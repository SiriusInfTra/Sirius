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
#include <glog/logging.h>
#include <random>

#include <server/resource_manager.h>
#include <server/infer_model_store.h>
#include <server/train_launcher.h>
#include <server/controller.h>
#include <server/profiler.h>
#include <server/config.h>


namespace colserve {

std::unique_ptr<TrainLauncher> TrainLauncher::train_launcher_;

std::pair<double, double> TrainLauncher::GetModelMemParam() {
  if (Config::use_shared_tensor_train) {
    if (cur_model_name_ == "resnet152") {
      /*
          8   1.50
         32   3.75
        128  11.75
        150  13.50
      
        AFTER EMPTY CACHE: 0.81 ~ 1.22
      */
      // return {1150, 85};
      return {1318, 145};
    } else {
      LOG(FATAL) << "Unsupported model: " << cur_model_name_;
    }
  } else {
    /*
        8     2.64
        32    4.97
        128  13.75
        150  15.68

        AFTER EMPTY CACHE: 2.28 ~ 2.37
    */
    if (cur_model_name_ == "resnet152") {
      return {2396, 145};
    } else {
      LOG(FATAL) << "Unsupported model: " << cur_model_name_;
    }
  }
}

void TrainLauncher::Init(const std::filesystem::path &train_store_path) {
  train_launcher_ = std::make_unique<TrainLauncher>();

  for (auto train_script : std::filesystem::directory_iterator(train_store_path)) {
    if (train_script.is_regular_file()) {
      train_launcher_->train_handles_[train_script.path().stem().string()] = train_script.path();
      LOG(INFO) << "TrainLauncher: Add " << train_script.path().stem().string();
    }
  }

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, 2);
  train_launcher_->thread_.reset(new std::thread([&]() {
    pthread_barrier_wait(&barrier);
    LOG(INFO) << "TrainLauncher thread start";
    while (Config::running) {
      train_launcher_->Train();
    }
  }));
  pthread_barrier_wait(&barrier);

  if (Config::dummy_adjust) {
    train_launcher_->dummy_adjust_thread_ = std::make_unique<std::thread>(
        &TrainLauncher::DummyAdjust, train_launcher_.get());
  }

  LOG(INFO) << "TrainLauncher initialized"; 
}

bool TrainLauncher::Shutdown() {
  if (train_launcher_->train_pid_ != -1) {
    LOG(INFO) << "[TrainLauncher]: Shutdown, train_pid(" << train_launcher_->train_pid_ << ") = -1.";
    CHECK_EQ(kill(train_launcher_->train_pid_, SIGKILL), 0);
    waitpid(train_launcher_->train_pid_, NULL, 0);
  }
  return true;
}

bool TrainLauncher::AddJob(network::TrainHandler::TrainData* data) {
  job_queue_.Put(std::make_shared<TrainJob>(data));
  return true;
}

double TrainLauncher::PredictMemUsageMB() {
  // LOG(INFO) << "Predict train memory, target batch size " << target_batch_size_;
  if (target_batch_size_ <= 0) {
    return 0;
  } else {
    auto [base, slope] = GetModelMemParam();
    return base + slope * target_batch_size_;
  }
}

int TrainLauncher::PredictTargetBatchSize(double memory_mb) {
  auto [base, slope] = GetModelMemParam();
  auto ret = static_cast<int>((memory_mb - base) / slope);
  ret = std::min(ret, job_batch_size_);
  ret = std::max(ret, 0);
  // LOG(INFO) << "## " << memory_mb << " " << base << " " << slope
  //           << " " << job_batch_size_;
  return ret;
}

int TrainLauncher::GetAdjustBatchSize(double memory_mb) {
  auto [base, slope] = GetModelMemParam();
  return static_cast<int>(std::ceil(memory_mb / slope));
}

bool TrainLauncher::Train() {
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
    // LOG(INFO) << "@@ " << arg_k << " " << arg_v;
    if (arg_k.find("batch-size") != std::string::npos) {
      job_batch_size_ = std::stoi(arg_v);
      cur_batch_size_ = job_batch_size_;
      target_batch_size_ = job_batch_size_;
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
      args_str.push_back("xsched-sync2");
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

bool TrainLauncher::LaunchTrain(std::shared_ptr<Job> job, std::vector<std::string> &args_str) {
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
    LOG(INFO) << "[TrainLauncher]: train wait infer idle in switch mode";
    Controller::Get()->WaitInferIdle();
  }
  LOG(INFO) << "fork train, batch size " << job_batch_size_;

  auto pid = fork();
  train_pid_ = pid;
  CHECK_GE(pid, 0) << "[TrainLauncher]: fork failed";

  if (pid == 0) {
    close(to_child_pipe[1]);
    dup2(to_child_pipe[0], STDIN_FILENO);
    if (Config::capture_train_log) {
      close(from_child_pipe[0]);
      dup2(from_child_pipe[1], STDOUT_FILENO);
    }
    if (Config::use_xsched) {
      auto xsched_path = Config::binary_directory / "xsched";
      auto xsched_lib_path = xsched_path / "lib";
      auto xsched_lib_cuda = xsched_lib_path / "libcuda.so.1";
      CHECK(std::filesystem::exists(xsched_lib_cuda));

      // std::string xsched_preload_path = xsched_path + "lib/libinstrument_sm70.so";
      
      CHECK_NE(setenv("LD_LIBRARY_PATH", xsched_lib_path.string().c_str(), 1), -1);
      extra_env_ss << "LD_LIBRARY_PATH=" << xsched_lib_path.string() << " ";
      // CHECK_NE(setenv("LD_PRELOAD", xsched_preload_path.c_str(), 1), -1);
      LOG(INFO) << "[TrainLauncher]: enable xsched.";
    }

    CHECK_NE(setenv("HAS_INFER_SERVER", "1", 1), -1);
    extra_env_ss << "HAS_INFER_SERVER=1" << " ";
    if (Config::use_shared_tensor_train) {
      CHECK_NE(setenv("USE_SHARED_TENSOR", "1", 1), -1);
      CHECK_NE(setenv("HAS_SHARED_TENSOR_SERVER", "1", 1), -1);
      CHECK_NE(setenv("SHARED_TENSOR_POOL_GB", std::to_string(Config::cuda_memory_pool_gb).c_str(), 1), -1);
      CHECK_NE(setenv("SHARED_TENSOR_POOL_FREELIST_POLICY", Config::mempool_freelist_policy.c_str(), 1), -1);
      extra_env_ss << "USE_SHARED_TENSOR=1"
                   << " HAS_SHARED_TENSOR_SERVER=1"
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
      LOG(INFO) << "[TrainLauncher]: set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to " << Config::train_mps_thread_percent;
    }

    LOG(INFO) << "[TrainLauncher]: " << "Train " << job << " ( "
              << extra_env_ss.str() << " " << ss.str() << ")";
    auto err = execvp("python", argv);
    // auto err = execvp("nsys", argv);
    perror("execvp");
    CHECK_GE(err, 0) << "[TrainLauncher]: spawn train worker fail, errno " << err;
  } else {
    close(to_child_pipe[0]);
    if (Config::capture_train_log) {
      close(from_child_pipe[1]);
    }
    // train_running_ = true;
    // 
    LOG(INFO) << "[TrainLauncher]: " << "Train " << job << " pid " << pid;
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
  int ret = waitpid(pid, &status, 0);
  LOG(INFO) << "[TrainLauncher]: wait pid return, ret = " << ret << ", status = " << status << "."; 
  train_pid_ = -1;
  batch_start_ = false;
  // target_batch_size_ = -1;
  // cur_batch_size_ = -1;
  Controller::Get()->TrainEnd(); // double check train end
  
  // LOG(INFO) << "signaled " << WIFSIGNALED(status) << " " << WTERMSIG(status);
  if (WIFSIGNALED(status)) {
    auto signal = WTERMSIG(status);
    if (signal == SIGKILL) {
      LOG(INFO) << "[TrainLauncher]: " << job << " is killed, restart";
      return false;
    } else {
      LOG(FATAL) << "[TrainLauncher]: " << job 
                 << " failed, signal is " << strsignal(signal) 
                 << " target_batch_size " << target_batch_size_ 
                 << " cur_batch_size " << cur_batch_size_ 
                 << " memory " << ResourceManager::GetTrainMemoryMB() << "MB"
                 << " predict memory " << PredictMemUsageMB() << "MB"
                 << " calculated free memory " << ResourceManager::GetFreeMemoryMB() << "MB";
      return false;
    }
  } else {
    return true;
  }
}

void TrainLauncher::DummyAdjust() {
  while (this->train_pid_ == -1) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::this_thread::sleep_for(std::chrono::seconds(15));
  LOG(INFO) << "DummyAdjust: train pid " << this->train_pid_;
  
  std::mt19937 gen(42); // fix seed

  auto start = Profiler::Now();
  while (Profiler::MilliFrom(start) < 30*1000 && this->train_pid_ != -1) {
    LOG(INFO) << "DummyAdjust at " << Profiler::GetTimeStamp();
    auto batch_size = 1;
    auto cmd_id = Controller::Get()->ColocateAdjust(-1, batch_size);
    Controller::Get()->WaitColocateAdjustDone(cmd_id);
    Controller::Get()->InferExit();
    std::this_thread::sleep_for(std::chrono::milliseconds(std::uniform_int_distribution<>(200, 1000)(gen)));
  }
}

} // namespace colserve