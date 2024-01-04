#include "logging_as_glog.h"
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>
#include "model_infer_store.h"
#include <glog/logging.h>

#include "model_train_store.h"
#include "controller.h"
#include "profiler.h"
#include "config.h"


namespace colserve {

std::unique_ptr<ModelTrainStore> ModelTrainStore::model_train_store_;

void ModelTrainStore::Init(const std::filesystem::path &train_store_path) {
  model_train_store_ = std::make_unique<ModelTrainStore>();

  for (auto train_script : std::filesystem::directory_iterator(train_store_path)) {
    if (train_script.is_regular_file()) {
      model_train_store_->train_handles_[train_script.path().stem().string()] = train_script.path();
      LOG(INFO) << "ModelTrainStore: Add " << train_script.path().stem().string();
    }
  }

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, 2);
  model_train_store_->thread_.reset(new std::thread([&]() {
    pthread_barrier_wait(&barrier);
    LOG(INFO) << "ModelTrainStore thread start";
    while (Config::running) {
      model_train_store_->Train();
    }
  }));
  pthread_barrier_wait(&barrier);

  LOG(INFO) << "ModelTrainStore initialized"; 
}

bool ModelTrainStore::Shutdown() {
  if (model_train_store_->train_pid_ != -1) {
    CHECK_EQ(kill(model_train_store_->train_pid_, SIGKILL), 0);
    waitpid(model_train_store_->train_pid_, NULL, 0);
  }
  return true;
}

bool ModelTrainStore::AddJob(network::TrainHandler::TrainData* data) {
  job_queue_.Put(std::make_shared<TrainJob>(data));
  return true;
}

double ModelTrainStore::PredictMemUsageMB() {
  if (train_pid_ == -1 || cur_batch_size_ <= 0) {
    return 0;
  } else {
    if (cur_model_name_ == "resnet152") {
      return cur_batch_size_ * 145;
    } else {
      LOG(FATAL) << "Unsupported model: " << cur_model_name_;
    }
  }
}

bool ModelTrainStore::Train() {
  std::shared_ptr<Job> job = nullptr;
  while (job == nullptr) {
    job = job_queue_.Get();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto data = job->GetTrainData();
  auto model = data->GetModelName();
  cur_model_name_ = model;
  auto train_script = train_handles_[model];

  std::vector<std::string> args_str;
  // args_str.push_back("nsys");
  // args_str.push_back("profile");
  
  args_str.push_back("python");
  args_str.push_back(train_script.string());
  auto args = data->GetTrainArgs();
  for (size_t i = 0, j = 0; i < args.size(); ) {
    std::string arg_k, arg_v;
    for (j = i; j < args.size() && args[j] != '='; j++);
    args_str.push_back("--" + args.substr(i, j - i)); 
    arg_k = args.substr(i, j - 1);
    for (i = j + 1, j = i; j < args.size() && args[j] != ' ' && args[j] != ','; j++); 
    args_str.push_back(args.substr(i, j - i));
    arg_v = args.substr(i, j - i);
    if (arg_k == "batch-size") {
      SetCurBatchSize(std::stoi(arg_v));
    }
    for (i = j + 1; i < args.size() && args[i] == ' '; i++);
  }
  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    args_str.push_back("--train-mode");
    args_str.push_back("taskswitch-l1");
  } else if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    args_str.push_back("--train-mode");
    args_str.push_back("taskswitch-l3");
  } else if (Config::serve_mode == ServeMode::kColocateL1) {
    args_str.push_back("--train-mode");
    args_str.push_back("colocate-l1");
  } else if (Config::serve_mode == ServeMode::kColocateL2) {
    args_str.push_back("--train-mode");
    args_str.push_back("colocate-l2");
  } else if (Config::serve_mode == ServeMode::kNormal) {
    args_str.push_back("--train-mode");
    args_str.push_back("normal");
  } else {
    LOG(FATAL) << "Unsupported serve mode: " << static_cast<int>(Config::serve_mode);
  }

  if (Config::serve_mode == ServeMode::kColocateL1 || Config::serve_mode == ServeMode::kTaskSwitchL1) {
    if (Config::use_xsched) {
      args_str.push_back("--hook-mode");
      args_str.push_back("xsched-sync");
    } else {
      args_str.push_back("--hook-mode");
      args_str.push_back("sync");
    }
  } else {
    args_str.push_back("--hook-mode");
    args_str.push_back("none");
  }

  if (Config::use_xsched) {
    args_str.push_back("--use-xsched");
    args_str.push_back("1");
  } else {
    args_str.push_back("--use-xsched");
    args_str.push_back("0");
  }

  args_str.push_back("--train-profile");
  args_str.push_back(Config::train_profile);

  while (Config::running) {
    if (LaunchTrain(job, args_str)) {
      break;
    }
  }

  LOG(INFO) << "Train: " << job << " finished";

  data->SetResult("train ok");
  data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
  return true;
}

bool ModelTrainStore::LaunchTrain(std::shared_ptr<Job> job, std::vector<std::string> &args_str) {
  std::stringstream extra_env_ss;
  std::stringstream ss;
  char* argv[args_str.size() + 1];
  for (size_t i = 0; i < args_str.size(); i++) {
    argv[i] = args_str[i].data();
    ss << args_str[i] << " ";
  }
  argv[args_str.size()] = 0;

  int from_child_pipe[2], to_child_pipe[2];
  CHECK_NE(pipe(from_child_pipe), -1);
  CHECK_NE(pipe(to_child_pipe), -1);

  if (Config::IsSwitchMode()) {
    LOG(INFO) << "[ModelTrainStore]: train wait infer idle in switch mode";
    Controller::Get()->WaitInferIdle();
  }
  LOG(INFO) << "fork train";

  auto pid = fork();
  train_pid_ = pid;
  CHECK_GE(pid, 0) << "[ModelTrainStore]: fork failed";

  if (pid == 0) {
    close(to_child_pipe[1]);
    dup2(to_child_pipe[0], STDIN_FILENO);
    if (Config::capture_train_log) {
      close(from_child_pipe[0]);
      dup2(from_child_pipe[1], STDOUT_FILENO);
    }
    if (Config::use_xsched) {
      std::string xsched_path = (Config::binary_directory / "xsched").string();
      std::string xsched_lib_path = xsched_path + "/lib";
      // std::string xsched_preload_path = xsched_path + "lib/libinstrument_sm70.so";
      
      CHECK_NE(setenv("LD_LIBRARY_PATH", xsched_lib_path.c_str(), 1), -1);
      extra_env_ss << "LD_LIBRARY_PATH=" << xsched_lib_path << " ";
      // CHECK_NE(setenv("LD_PRELOAD", xsched_preload_path.c_str(), 1), -1);
      LOG(INFO) << "[ModelTrainStore]: enable xsched.";
    }

    if (Config::use_shared_tensor_train) {
      CHECK_NE(setenv("USE_SHARED_TENSOR", "1", 1), -1);
      CHECK_NE(setenv("SHARED_TENSOR_HAS_SERVER", "1", 1), -1);
      CHECK_NE(setenv("SHARED_TENSOR_POOL_GB", std::to_string(Config::cuda_memory_pool_gb).c_str(), 1), -1);
      CHECK_NE(setenv("SHARED_TENSOR_POOL_FREELIST_POLICY", Config::mempool_freelist_policy.c_str(), 1), -1);
      extra_env_ss << "USE_SHARED_TENSOR=1"
                   << " SHARED_TENSOR_HAS_SERVER=1"
                   << " SHARED_TENSOR_POOL_GB=" << Config::cuda_memory_pool_gb
                   << " SHARED_TENSOR_POOL_FREELIST_POLICY=" << Config::mempool_freelist_policy << " ";
      // CHECK_NE(setenv("CUDA_LAUNCH_BLOCKING", "1", 1), -1);
    } else {
      CHECK_NE(setenv("USE_SHARED_TENSOR", "0", 1), -1);
      extra_env_ss << "USE_SHARED_TENSOR=0 ";
    }
    if (Config::train_mps_thread_percent >= 0 && Config::train_mps_thread_percent <= 100) {
      CHECK_NE(setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", std::to_string(Config::train_mps_thread_percent).c_str(), 1), -1);
      extra_env_ss << "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=" << Config::train_mps_thread_percent << " ";
      LOG(INFO) << "[ModelTrainStore]: set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to " << Config::train_mps_thread_percent;
    }

    LOG(INFO) << "[ModelTrainStore]: " << "Train " << job << " ( "
              << extra_env_ss.str() << " " << ss.str() << ")";
    auto err = execvp("python", argv);
    // auto err = execvp("nsys", argv);
    perror("execvp");
    CHECK_GE(err, 0) << "[ModelTrainStore]: spawn train worker fail, errno " << err;
  } else {
    close(to_child_pipe[0]);
    if (Config::capture_train_log) {
      close(from_child_pipe[1]);
    }
    // train_running_ = true;
    LOG(INFO) << "[ModelTrainStore]: " << "Train " << job << " pid " << pid;
  }

  if (Config::capture_train_log) {
    std::array<char, 1024> buf;
    auto fp = fdopen(from_child_pipe[0], "r");
    while (fgets(buf.data(), buf.size(), fp)) {
      LOG(INFO) << "[PyTorch backend] Train: " << buf.data();
    }
    fclose(fp);
    close(from_child_pipe[0]);
  }
  close(to_child_pipe[1]);

  int status;
  waitpid(pid, &status, 0);
  train_pid_ = -1;
  cur_batch_size_ = -1;
  Controller::Get()->TrainEnd(); // double check train end
  
  // LOG(INFO) << "signaled " << WIFSIGNALED(status) << " " << WTERMSIG(status);
  if (WIFSIGNALED(status)) {
    auto signal = WTERMSIG(status);
    if (signal == SIGKILL) {
      LOG(INFO) << "[ModelTrainStore]: " << job << " is killed, restart";
      return false;
    } else {
      LOG(FATAL) << "[ModelTrainStore]: " << job << " failed, signal is " << strsignal(signal)
                 << " cur_batch_size " << cur_batch_size_ << " predict memory " << PredictMemUsageMB() << "MB";
      return false;
    }
  } else {
    return true;
  }
}

} // namespace colserve