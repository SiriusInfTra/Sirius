#include <common/dtype_helper.h>
#include <server/logging_as_glog.h>
#include <server/infer_model_store.h>
#include <server/infer_model.h>
#include <server/train_launcher.h>
#include <server/controller.h>
#include <server/profiler.h>
#include <server/config.h> 

#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <regex>


namespace colserve {
namespace {
std::vector<std::string> ParseModelName(const std::string &model) {
  std::regex r{"([a-zA-Z0-9_]+)(\\[([0-9]+)\\])?"};
  std::smatch match;
  CHECK(std::regex_match(model, match, r)) << "model name " << model << " not match";
  CHECK_EQ(match.size(), 4);
  if (match[3].str().empty()) {
    return {match[1].str()};
  } else {
    CHECK_GE(std::stoi(match[3].str()), 1);
    std::vector<std::string> ret{match[1].str()};
    for (int i = 1; i < std::stoi(match[3].str()); i++) {
      ret.push_back(match[1].str() + "-" + std::to_string(i));
    }
    return ret;
  }
}
}

std::unique_ptr<InferModelStore> InferModelStore::infer_model_store_;

InferModelStore* InferModelStore::Get() {
  if (infer_model_store_ == nullptr) {
    LOG(FATAL) << "InferModelStore not initialized";
  }
  return infer_model_store_.get();
}

void InferModelStore::Init(const std::filesystem::path &infer_store_path) {
  LOG(INFO) << "InferModelStore start initializing";

  infer_model_store_ = std::make_unique<InferModelStore>();
  infer_model_store_->models_["dummy"] = std::make_unique<Model>();

  // model_name -> [key -> value]
  std::map<std::string, std::map<std::string, std::string>> models;
  
  std::filesystem::path config_file_path;
  if (Config::infer_model_config_path.at(0) == '/') {
    config_file_path = Config::infer_model_config_path;
  } else {
    config_file_path = infer_store_path / Config::infer_model_config_path;
  }

  std::ifstream config_file(config_file_path);
  if (config_file.good()) {
    std::string cur_model;
    for (std::string line; std::getline(config_file, line);) {
      if (line.empty()) continue;
      if (line[0] == '#') continue;
      std::istringstream iss(line);
      if (line[0] != ' ' && line[0] != '\n' && line[0] != '\t') {
        iss >> cur_model;
        models[cur_model] = {};
      } else {
        std::string key, value;
        iss >> key >> value;
        models[cur_model][key] = value;
      }
    }
    for (auto &model: models) {
      std::stringstream ss;
      ss << "[" << model.first << "] "; 
      CHECK(model.second.count("path"));
      CHECK(model.second.count("device"));
      CHECK(model.second.count("batch-size"));
      if (!model.second.count("num-worker")) {
        model.second["num-worker"] = "1";
      }
      if (!model.second.count("max-worker")) {
        model.second["max-worker"] = "1";
      }
      for (auto &kv : model.second) {
        ss << kv.first << "=" << kv.second << " ";
      }
      LOG(INFO) << "[InferModelStore] Read from config file: " << ss.str();
    }
  }

  // mod.so, mod.json, mod.params should be in the model directory
  for (auto &model : models) {
    auto model_path = infer_store_path / model.second["path"];
    CHECK(std::filesystem::exists(model_path)) << "InferModelStore: " << model_path << " not exist";
    CHECK(std::filesystem::is_directory(model_path)) << model_path << " is not a directory";
    auto model_params = tvm::TVMGraph::LoadParamsAsTVMArray(
        (model_path / "mod.params").c_str());
    
    for (auto model_name : ParseModelName(model.first)) {
      auto model_device = model.second["device"];
      size_t batch_size = std::stoi(model.second["batch-size"]);
      DLDevice device;
      if (model_device == "cuda") {
        device = DLDevice{kDLCUDA, 0};
      } else {
        LOG(FATAL) << model_name << " unsupport device type " << model_device;
      }
      infer_model_store_->models_[model_name] = 
          std::make_unique<Model>(model_name, model_path, model_params,
                                  device, batch_size, 
                                  std::stoi(model.second["num-worker"]),
                                  std::stoi(model.second["max-worker"]));
    }
    LOG_IF(INFO, Config::log_model_init_info) 
        << "[InferModelStore Init] "<< "Add " << model.first << ":" << model.second["device"]
        << ", batch-size=" << model.second["batch-size"];
  }

  // if (Config::IsSwitchMode()) {
  //   infer_model_store_->task_switch_control_.reset(new std::thread([&]() {
  //     while (true) {
  //       if (Controller::Get()->IsInferIdle()) {
  //         // first ensure all not more infer workers
  //         std::unique_lock lock{infer_model_store_->task_switch_mutex_};
  //         InferModelStore::Get()->task_switch_cv.wait(lock, [&]() {
  //           return infer_model_store_->task_switch_enter_cnt_ > 0 
  //               && (infer_model_store_->task_switch_control_cnter_ == 
  //                   static_cast<int>(InferModelStore::TaskSwitchStatus::kNotAddWorker));
  //         });

  //         pthread_barrier_init(&infer_model_store_->task_switch_barrier, nullptr, 
  //             infer_model_store_->task_switch_enter_cnt_ + 1);
  //         // enter task switch prepare exit stage
  //         infer_model_store_->task_switch_control_cnter_ = static_cast<int>(InferModelStore::TaskSwitchStatus::kPrepareExit);
  //         // LOG(INFO) << "[InferModelStore]: try task switch " << infer_model_store_->task_switch_enter_cnt_;
  //         auto t0 = Profiler::Get()->Now();

  //         // let all infer enter task switch prepare exit stage
  //         pthread_barrier_wait(&infer_model_store_->task_switch_barrier);
  //         auto wait_task_exit_ms = Profiler::Get()->MilliFrom(t0);
  //         LOG(INFO) << "[InferModelStore] [Task Switch]: wait for inference threads " << wait_task_exit_ms << " ms "
  //                   << " wait up to " << Config::task_switch_delay_ms << " ms";
  //         if (wait_task_exit_ms < Config::task_switch_delay_ms) {
  //           auto delay_us = static_cast<int>((Config::task_switch_delay_ms - wait_task_exit_ms) * 1000);
  //           std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  //         }
  //         if (Controller::Get()->IsInferIdle()) {
  //           infer_model_store_->task_switch_control_cnter_ = static_cast<int>(InferModelStore::TaskSwitchStatus::kExit);
  //           Profiler::Get()->RecordPerf(Profiler::PerfItem::InferNumModelOnSwitch, infer_model_store_->task_switch_enter_cnt_);
  //           // all infer know result
  //           pthread_barrier_wait(&infer_model_store_->task_switch_barrier);
  //           LOG(INFO) << "[InferModelStore] [Task Switch]: task switch to train | " << Controller::Get()->GetInferStatusStr();
  //           // all infer do task switch
  //           pthread_barrier_wait(&infer_model_store_->task_switch_barrier);
  //           Controller::Get()->ResumeTrain();
  //         } else {
  //           infer_model_store_->task_switch_control_cnter_ = static_cast<int>(InferModelStore::TaskSwitchStatus::kCancelExit);
  //           // all infer know result
  //           pthread_barrier_wait(&infer_model_store_->task_switch_barrier);
  //           // all infer cancel task switch
  //           pthread_barrier_wait(&infer_model_store_->task_switch_barrier); 
  //           LOG(INFO) << "[InferModelStore] [Task Switch]: task switch cancel exit | " << Controller::Get()->GetInferStatusStr();
  //         }
  //         infer_model_store_->task_switch_control_cnter_ = static_cast<int>(InferModelStore::TaskSwitchStatus::kNotAddWorker);
  //         infer_model_store_->task_switch_cv.notify_all();
  //         pthread_barrier_destroy(&infer_model_store_->task_switch_barrier);
  //       } else {
  //         // Controller::Get()->LogInferStatus();
  //       }
  //       std::this_thread::sleep_for(std::chrono::milliseconds(1));
  //     }
  //   }));
  // }

  if (Config::IsColocateMode()) {
    infer_model_store_->monitor_thread_.reset(
        new std::thread{&InferModelStore::ColocateMonitor, infer_model_store_.get()});
  } else if (Config::IsSwitchMode()) {
    infer_model_store_->monitor_thread_.reset(
        new std::thread{&InferModelStore::TaskSwitchMonitor, infer_model_store_.get()});
  }
  
  infer_model_store_->warmup_done_ = !Config::has_warmup;
  LOG(INFO) << "InferModelStore initialized";
}

void InferModelStore::WarmupDone() {
  infer_model_store_->warmup_done_ = true;
  LOG(INFO) << "[InferModelStore] warmup done, num ready model "
            << Model::GetNumModel(Model::Status::kReady);
}

void InferModelStore::InferingInc() {
  std::unique_lock lock{Get()->task_switch_mutex_, std::defer_lock};
  if (Config::IsSwitchMode()) {
    lock.lock();
  }
  // Get()->task_switch_cv_.wait(lock, []() {
  //   return Get()->task_switch_ctrl_.load() != static_cast<int>(TaskSwitchStatus::kReclaimInfer);
  // });
  auto res = Get()->num_infering_model_.fetch_add(1, std::memory_order_relaxed);
  // if (res == 0) {
  //   Get()->task_switch_ctrl_.store(static_cast<int>(TaskSwitchStatus::kInfering));
  // }
}

void InferModelStore::InferingDec() {
  // std::unique_lock lock{Get()->task_switch_mutex_};
  // CHECK(Get()->task_switch_ctrl_.load() == static_cast<int>(TaskSwitchStatus::kInfering));
  CHECK(Get()->num_infering_model_.load(std::memory_order_relaxed) > 0);
  auto res = Get()->num_infering_model_.fetch_sub(1, std::memory_order_relaxed);
  // if (res == 1) {
  //   Get()->task_switch_ctrl_.store(static_cast<int>(TaskSwitchStatus::kNotInfering));
  // }
}

Model* InferModelStore::GetModel(const std::string &name) {
  auto it = models_.find(name);
  if (it == models_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

size_t InferModelStore::NumJobs() {
  size_t num_jobs = 0;
  for (auto &model : models_) {
    num_jobs += model.second->NumJobs();
  }
  return num_jobs;
}

void InferModelStore::ColocateMonitor() {
  using namespace std::chrono_literals;
  while (true) {
    int num_exit = 0;
    for (auto& [model_name, model] : models_) {
      if (model->GetIdleMill(0) > GetMaxIdleMill()) {
        bool res = model->ReclaimMemory(0);
        if (res) {
          num_exit++;
          LOG(INFO) << "[InferModelStore] " << model_name << " reclaim memory";
        }
      }
    }
    if (num_exit > 0) Controller::Get()->InferExit();
    std::this_thread::sleep_for(10ms);
  }
}

void InferModelStore::TaskSwitchMonitor() {
  using namespace std::chrono_literals;

  auto check_switch = [this]() {
    return this->num_infering_model_.load(std::memory_order_relaxed) == 0
        && Controller::Get()->IsInferIdle();
  };

  auto do_switch = [this, &check_switch]() -> int {
    std::unique_lock lock{this->task_switch_mutex_};
    if (!check_switch()) { return -1; }

    int reclaim_cnt = 0;
    for (auto &model : this->models_) {
      reclaim_cnt += model.second->ReclaimMemory(0);
    }
    return reclaim_cnt;
  };

  while (true) {
    if (check_switch()) {
      auto prepare_switch_time = Profiler::Get()->Now();
      std::this_thread::sleep_for(Config::task_switch_delay_ms * 1ms);
      
      int res = do_switch();
      if (res < 0) {
        LOG(INFO) << "[InferModelStore] task switch cancelled";
      } else {
        LOG(INFO) << "[InferModelStore] task switch done, reclaim " << res << " models";
        Profiler::Get()->RecordPerf(Profiler::PerfItem::InferNumModelOnSwitch, res);
        Controller::Get()->ResumeTrain();
      }
    }
  }
}

} // namespace colserve