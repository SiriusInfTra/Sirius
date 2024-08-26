#include <server/logging_as_glog.h>
#include <server/resource_manager.h>
#include <server/model_store/infer_model_store.h>
#include <server/train_launcher.h>
#include <server/control/controller.h>
#include <server/train_adjuster.h>
#include <server/profiler.h>
#include <server/config.h>

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>

#include <boost/range/irange.hpp>
#include <boost/range/numeric.hpp>

#include <glog/logging.h>
#include <random>
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <chrono>


namespace colserve {

std::unique_ptr<TrainLauncher> TrainLauncher::train_launcher_;

void TrainLauncher::Init(const std::filesystem::path &train_store_path) {
  train_launcher_ = std::make_unique<TrainLauncher>();

  for (auto train_script : std::filesystem::directory_iterator(train_store_path)) {
    if (train_script.is_regular_file()) {
      train_launcher_->train_handles_[train_script.path().stem().string()] = train_script.path();
      LOG_IF(INFO, Config::log_train_init) << "TrainLauncher: Add " 
                                           << train_script.path().stem().string();
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
  train_launcher_->KillTrain();
  return true;
}

bool TrainLauncher::AddJob(network::TrainHandler::TrainData* data) {
  job_queue_.Put(std::make_shared<TrainJob>(data));
  return true;
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

  if ((Config::IsColocateMode() || Config::IsSwitchMode()) && 
      Config::use_shared_tensor &&
      Config::enable_warm_cache_fallback) {
    auto [base, slope] = TrainAdjuster::adjuster_->GetModelMemParam(cur_model_name_);
    Config::max_warm_cache_nbytes = static_cast<size_t>((
        Config::cuda_memory_pool_gb * 1024 
        - Config::train_memory_over_predict_mb - base
      ) * 1_MB);
    LOG(INFO) << "[Warm Cache Fallback for Colocation] set max warm cache nbytes to "
              << sta::PrintByte(Config::max_warm_cache_nbytes);
  }

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

  // args_str.push_back("--train-profile");
  // args_str.push_back(Config::train_profile);

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

bool TrainLauncher::LaunchTrain(std::shared_ptr<Job> job, 
                                std::vector<std::string> &args_str) {
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
    ctrl::Controller::Get()->WaitInferIdle();
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
      CHECK(std::filesystem::exists(xsched_lib_cuda)) << xsched_lib_cuda << " not exists";

      // std::string xsched_preload_path = xsched_path + "lib/libinstrument_sm70.so";
      CHECK_NE(setenv("COL_USE_XSCHED", "1", 1), -1);
      extra_env_ss << "COL_USE_XSCHED=1" << " ";
      CHECK_NE(setenv("LD_LIBRARY_PATH", xsched_lib_path.string().c_str(), 1), -1);
      extra_env_ss << "LD_LIBRARY_PATH=" << xsched_lib_path.string() << " ";
      // CHECK_NE(setenv("LD_PRELOAD", xsched_preload_path.c_str(), 1), -1);
      LOG(INFO) << "[TrainLauncher]: enable xsched.";
    } else {
      CHECK_NE(setenv("COL_USE_XSCHED", "0", 1), -1);
      extra_env_ss << "COL_USE_XSCHED=0" << " ";
    }

    CHECK_NE(setenv("COL_HAS_INFER_SERVER", "1", 1), -1);
    extra_env_ss << "COL_HAS_INFER_SERVER=1" << " ";
    if (Config::use_shared_tensor_train) {
      CHECK_NE(setenv("COL_USE_SHARED_TENSOR", "1", 1), -1);
      CHECK_NE(setenv("COL_HAS_SHARED_TENSOR_SERVER", "1", 1), -1);
      CHECK_NE(setenv("COL_SHARED_TENSOR_POOL_GB", 
                      std::to_string(Config::cuda_memory_pool_gb).c_str(), 1), -1);
      CHECK_NE(setenv("SHARED_TENSOR_POOL_FREELIST_POLICY", 
                      Config::mempool_freelist_policy.c_str(), 1), -1);
      extra_env_ss << "COL_USE_SHARED_TENSOR=1"
                   << " COL_HAS_SHARED_TENSOR_SERVER=1"
                   << " COL_SHARED_TENSOR_POOL_GB=" 
                   << Config::cuda_memory_pool_gb
                   << " SHARED_TENSOR_POOL_FREELIST_POLICY=" 
                   << Config::mempool_freelist_policy 
                   << " ";
      // CHECK_NE(setenv("CUDA_LAUNCH_BLOCKING", "1", 1), -1);
    } else {
      CHECK_NE(setenv("COL_USE_SHARED_TENSOR", "0", 1), -1);
      extra_env_ss << "COL_USE_SHARED_TENSOR=0 ";
    }

    if (Config::serve_mode == ServeMode::kColocateL1 
        || Config::serve_mode == ServeMode::kTaskSwitchL1) {
      if (Config::use_xsched) {
        CHECK_NE(setenv("COL_HOOK_MODE", "xsched-sync2", 1), -1);
        extra_env_ss << "COL_HOOK_MODE=xsched-sync2 ";
        // use xsched-sync for dummy adjust
      } else {
        CHECK_NE(setenv("COL_HOOK_MODE", "sync", 1), -1);
        extra_env_ss << "COL_HOOK_MODE=sync ";
      }
    } else {
      CHECK_NE(setenv("COL_HOOK_MODE", "none", 1), -1);
      extra_env_ss << "COL_HOOK_MODE=none ";
    }

    if (!Config::train_profile.empty()) {
      CHECK_NE(setenv("COL_TRAIN_PROFILE_LOG_PATH", 
                      Config::train_profile.c_str(), 1), -1);
      extra_env_ss << "COL_TRAIN_PROFILE_LOG_PATH=" << Config::train_profile << " ";
    }

    if (Config::skip_set_mps_thread_percent) {
      LOG(INFO) << "[TrainLauncher]: skip set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE";
      if (Config::dynamic_sm_partition) {
        CHECK_NE(setenv("COL_DYNAMIC_SM_PARTITION", "1", 1), -1);
        extra_env_ss << "COL_DYNAMIC_SM_PARTITION=1";
      }
    } else if (Config::train_mps_thread_percent >= 0 
               && Config::train_mps_thread_percent <= 100) {
      CHECK_NE(setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", 
                      std::to_string(Config::train_mps_thread_percent).c_str(), 1), -1);
      extra_env_ss << "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=" 
                   << Config::train_mps_thread_percent << " ";
      LOG(INFO) << "[TrainLauncher]: set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to " 
                << Config::train_mps_thread_percent;
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
  LOG(INFO) << "[TrainLauncher]: wait pid return, ret = " << ret 
            << ", status = " << status << "."; 
  train_pid_ = -1;
  batch_start_ = false;
  // target_batch_size_ = -1;
  // cur_batch_size_ = -1;
  ctrl::Controller::Get()->TrainEnd(); // double check train end
  
  // LOG(INFO) << "signaled " << WIFSIGNALED(status) << " " << WTERMSIG(status);
  if (WIFSIGNALED(status)) {
    auto signal = WTERMSIG(status);
    if (signal == SIGKILL) {
      LOG(INFO) << "[TrainLauncher]: " << job << " is killed, restart";
      return false;
    } else {
      int train_world_size = ctrl::InfTraCommunicator::GetTrainWorldSize();
      std::string train_memory_str, free_memory_str;
      for (auto i : boost::irange(train_world_size)) {
        train_memory_str += 
            std::to_string(ResourceManager::GetTrainMemoryMB(i)) + " ";
        free_memory_str += 
            std::to_string(ResourceManager::GetFreeMemoryMB(i, true)) + " ";
      }
      for (auto pid : ctrl::InfTraCommunicator::GetTrainPIDs()) {
        kill(pid, SIGKILL);
      }
      LOG(FATAL) << "[TrainLauncher]: " << job 
                 << " failed, signal is " << strsignal(signal) 
                 << " target_batch_size " << target_batch_size_ 
                 << " cur_batch_size " << cur_batch_size_ 
                 << " memory [ " << train_memory_str << "] MB"
                 << " predict memory " 
                 << TrainAdjuster::PredictTrainMemUsageMB(0, false) << "MB"
                 << " calculated free memory [ " << free_memory_str << "] MB";
      return false;
    }
  } else {
    return true;
  }
}

// used for eval issue of un-released tensor
void TrainLauncher::DummyAdjust() {
  while (this->train_pid_ == -1) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  size_t delay_before_dummy_adjust = 60;
  LOG(INFO) << "DummyAdjust: train pid " << this->train_pid_ 
            << " start after " << delay_before_dummy_adjust << "s";
  std::this_thread::sleep_for(std::chrono::seconds(delay_before_dummy_adjust));
  
  std::mt19937 gen(42); // fix seed

  int ori_target_bs = this->target_batch_size_;

  auto start = Profiler::Now();
  while (Profiler::MilliFrom(start) < 60*1000 && this->train_pid_ != -1) {
    LOG(INFO) << "DummyAdjust at " << Profiler::GetTimeStamp();
    auto batch_size = 1;
    auto cmd_id = ctrl::Controller::Get()->ColocateAdjust(-1, 0, batch_size);
    ctrl::Controller::Get()->WaitColocateAdjustDone(cmd_id);
    ctrl::Controller::Get()->DummyInferExit(0, ori_target_bs);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(std::uniform_int_distribution<>(200, 1000)(gen)));
  }
}

void TrainLauncher::KillTrain() {
  if (!ctrl::InfTraCommunicator::GetSinfo()->IsTrainInfoValid(ctrl::kTraRank_0)) {
    return;
  }
  int num_train_proc = 
      ctrl::InfTraCommunicator::GetSinfo()->GetTrainInfoUnsafe(0)->train_world_size;
  for (int i = 0; i < num_train_proc; i++) {
    auto train_info = ctrl::InfTraCommunicator::GetSinfo()->GetTrainInfoUnsafe(i);
    auto pid = train_info->train_pid;
    auto rank = train_info->train_rank;
    if (pid != -1) {
      LOG(INFO) << "[TrainLauncher]: Kill train pid " << pid << " rank " << rank;
      auto err = kill(pid, SIGKILL);
      CHECK(err == 0 || errno == ESRCH);
    }
  }
}

} // namespace colserve