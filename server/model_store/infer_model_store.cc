#include <server/logging_as_glog.h>
#include <server/model_store/infer_model_store.h>
#include <server/model_store/infer_model.h>
#include <server/model_store/model_cache.h>
#include <server/train_launcher.h>
#include <server/control/controller.h>
#include <server/train_adjuster.h>
#include <server/profiler.h>
#include <server/config.h> 

#include <common/tensor/dtype_helper.h>
#include <common/device_manager.h>

#include <boost/range/irange.hpp>

#include <atomic>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>
#include <mutex>
#include <regex>


namespace colserve {
namespace {
std::vector<std::string> ParseModelName(const std::string &model) {
  std::regex r{"([a-zA-Z0-9_]+)(\\[([0-9]+)\\])?"};
  std::smatch match;
  CHECK(std::regex_match(model, match, r)) 
      << "model name " << model << " not match";
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

std::unique_ptr<InferModelStore> InferModelStore::infer_model_store_ = nullptr;

InferModelStore* InferModelStore::Get() {
  if (infer_model_store_ == nullptr) {
    LOG(FATAL) << "InferModelStore not initialized";
  }
  return infer_model_store_.get();
}

void InferModelStore::Init(const std::filesystem::path &infer_store_path) {
  if (colserve::Config::no_infer) {
    LOG(INFO) << "InferModelStore skip initializing";
    return;
  }
  LOG(INFO) << "InferModelStore start initializing";

  WarmModelCache::Init();
  ColdModelCache::Init();
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
  int next_gpu = 0;
  for (auto &model : models) {
    auto model_path = infer_store_path / model.second["path"];
    CHECK(std::filesystem::exists(model_path)) 
        << "InferModelStore: " << model_path << " not exist";
    CHECK(std::filesystem::is_directory(model_path)) 
        << model_path << " is not a directory";

    auto model_params = tvm::TVMGraph::LoadParamsAsTVMArray(
        (model_path / "mod.params").c_str());
    
    for (auto model_name : ParseModelName(model.first)) {
      auto model_device = model.second["device"];
      size_t batch_size = std::stoi(model.second["batch-size"]);
      DLDevice device;
      if (model_device == "cuda") {
        device = DLDevice{kDLCUDA, next_gpu++};
      } else {
        LOG(FATAL) << model_name << " unsupport device type " << model_device;
      }
      infer_model_store_->models_[model_name] = 
          std::make_unique<Model>(model_name, model_path, model_params,
                                  device, batch_size, 
                                  std::stoi(model.second["num-worker"]),
                                  std::stoi(model.second["max-worker"]));
      if (Config::model_place_policy == ModelPlacePolicy::kRoundRobin) {
        if (next_gpu >= sta::DeviceManager::GetNumVisibleGpu()) {
          next_gpu = 0;
        }
      }
    }
    LOG_IF(INFO, Config::log_infer_model_init) 
        << "[InferModelStore Init] "<< "Add " << model.first << ":" << model.second["device"]
        << ", batch-size=" << model.second["batch-size"];
  }

  if (Config::IsColocateMode()) {
    infer_model_store_->monitor_thread_.reset(
        new std::thread{&InferModelStore::ColocateMonitor, infer_model_store_.get()});
  } else if (Config::IsSwitchMode()) {
    infer_model_store_->monitor_thread_.reset(
        new std::thread{&InferModelStore::TaskSwitchMonitor, infer_model_store_.get()});
  }
  
  infer_model_store_->warmup_done_ = !Config::has_warmup;
  infer_model_store_->initialized_ = true;
  LOG(INFO) << "InferModelStore initialized";
}

bool InferModelStore::Shutdown() {
  if (Config::no_infer) {
    LOG(INFO) << "InferModelStore skip shutdown";
    return true;
  }
  std::stringstream ss;
  ss << "\n[Model Hotness]: \n";
  int i = 0;
  for (auto & [model_name, model] : infer_model_store_->models_) {
    i++;
    ss << "[" << model_name << " " << model->GetHotness() << "] ";
    if (i % 5 == 0) {
      ss << "\n";
    }
  }
  LOG(INFO) << ss.str();

  return true; 
}

void InferModelStore::WarmupDone() {
  // Reclaim all models for colcoation mode and switch mode,
  // for static partition, we don't need to clear models.
  // As cold cache is zero for static partition, 
  // the result is same if we dnot clear cold cache for static partition.
  if (Config::IsColocateMode() || Config::IsSwitchMode()) {
    infer_model_store_->ClearWarmCache();
    infer_model_store_->ClearColdCache();
  }
  infer_model_store_->warmup_done_ = true;
  LOG(INFO) << "[InferModelStore] warmup done, num ready model "
            << Model::GetNumModel(Model::Status::kReady);
}

bool InferModelStore::AddJob(const std::string &model_name,
                             network::InferHandler::InferData* data) {
  if (model_name == "dummy") {
    data->GetResponse().set_result("dummy result");
    data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    return true;
  }

  auto model = infer_model_store_->GetModel(model_name);
  if (!model) {
    return false;
  }

  Profiler::InferReqInc();
  // InterruptTrain check whether to interrupt train
  if (Config::IsSwitchMode()) {
    auto cmd_id = ctrl::Controller::Get()->InterruptTrain();
    
    auto t0 = Profiler::Now();
    ctrl::Controller::Get()->WaitTaskSwitchDone(cmd_id);
    auto wait_train_ms = Profiler::MilliFrom(t0);

    LOG_IF(INFO, Config::log_task_switch && cmd_id != 0) 
        << "[InferModelStore] [Task Switch] wait train " 
        << wait_train_ms << " ms, cmd_id "
        << cmd_id;
  }

  {
    std::unique_lock lock{infer_model_store_->mutex_};
    infer_model_store_->queing_infer_reqs_.insert(data->GetId());
  }
  model->AddJob(data);
  return true;
}

void InferModelStore::InferingInc(tvm::TVMGraph *graph, tvm::Executor *executor) {
  std::unique_lock lock{Get()->task_switch_mutex_, std::defer_lock};
  if (Config::IsSwitchMode()) {
    lock.lock();
    Get()->task_switch_to_infer_ = true;
  }
  // Get()->task_switch_cv_.wait(lock, []() {
  //   return Get()->task_switch_ctrl_.load() != static_cast<int>(TaskSwitchStatus::kReclaimInfer);
  // });
  Get()->num_infering_model_.fetch_add(1, std::memory_order_relaxed);
  Get()->infering_model_nbytes_.fetch_add(graph->GetStorageNbytes(), std::memory_order_relaxed); 

  // if (res == 0) {
  //   Get()->task_switch_ctrl_.store(static_cast<int>(TaskSwitchStatus::kInfering));
  // }
}

void InferModelStore::InferingDec(tvm::TVMGraph *graph, tvm::Executor *executor) {
  // std::unique_lock lock{Get()->task_switch_mutex_};
  // CHECK(Get()->task_switch_ctrl_.load() == static_cast<int>(TaskSwitchStatus::kInfering));
  CHECK(Get()->num_infering_model_.load(std::memory_order_relaxed) > 0);
  Get()->num_infering_model_.fetch_sub(1, std::memory_order_relaxed);
  Get()->infering_model_nbytes_.fetch_sub(graph->GetStorageNbytes(), std::memory_order_relaxed);
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

void InferModelStore::ClearColdCache() {
  LOG(INFO) << "ClearColdCache";
  for (auto & cold_model_cache : ColdModelCache::cold_model_caches_) {
    auto cold_cache_lock = cold_model_cache->Lock();
    int rank = 0;
    for (auto &&[name, model]: models_) {
      auto &&[evict_groups_id, succ] = 
          cold_model_cache->PopCacheItem(name, rank, false, cold_cache_lock);
      if (succ) { 
        model->ClearColdCache(evict_groups_id, rank, cold_cache_lock); 
      }
    }
    CHECK_EQ(cold_model_cache->GetCachedNbytes(cold_cache_lock), 0);
    cold_model_cache->SetNewCapacity(0, cold_cache_lock);
  }
}

void InferModelStore::ClearWarmCache() {
  LOG(INFO) << "ClearWarmCache";
  if (!WarmModelCache::Enable()) {
    return ;
  }
  for (auto & warm_model_cache : WarmModelCache::warm_model_caches_) {
    CHECK(warm_model_cache != nullptr);
    auto warm_cache_lock = warm_model_cache->Lock();
    for (auto &&[name, cache_item]: warm_model_cache->warm_cache_items_) {
      auto model = cache_item->model;
      CHECK(model != nullptr);
      CHECK_EQ(model->num_worker_, 1);
      std::unique_lock warm_cache_model_lock{cache_item->mut};
      bool res = model->ReclaimMemory(0, model);

      // force let cache = false
      cache_item->cached = false;
    }
    warm_model_cache->cached_nbytes_ = 0;
  }
}

void InferModelStore::ColocateMonitor() {
  LOG(INFO) << "Start ColocateMonitor";
  using namespace std::chrono_literals;
  while (true) {
    // int num_exit = 0;
    std::array<int, MAX_DEVICE_NUM> num_exits{0};
    for (auto& [model_name, model] : models_) {
      if (model_name == "dummy") continue;
      if (model->GetIdleMill(0) > GetMaxIdleMill()) {
        const int rank = 0;
        const int device_id = model->device_.device_id;
        auto cold_cache_lock = ColdModelCache::Get(device_id)->Lock();
        std::unique_lock<std::mutex> model_lock{model->muts_[rank]};
        bool res = model->ReclaimMemory(rank, cold_cache_lock, model_lock, nullptr);
        if (res) {
          num_exits[device_id]++;
          auto cold_cache_nbytes = 
              ColdModelCache::Get(device_id)->GetCachedNbytes(cold_cache_lock);
          LOG_IF(INFO, Config::log_infer_model_reclaim) 
              << "[InferModelStore] reclaim " << model_name
              << ", cold cache nbytes "
              << sta::PrintByte(cold_cache_nbytes);
        }
      }
    }

    if (!ctrl::Controller::Get()->IsTrainIdle() 
        && std::accumulate(num_exits.begin(), num_exits.end(), 0) > 0) {
      auto adjust_plan = TrainAdjuster::GetInferReleaseMemAdjustPlan();
      if (!adjust_plan.empty()) {
        ctrl::Controller::Get()->ColocateInferReleaseAdjust(adjust_plan);
      }
    }
    std::this_thread::sleep_for(100ms);
  }
}

void InferModelStore::TaskSwitchMonitor() {
  using namespace std::chrono_literals;

  auto check_switch_unlock = [this](std::unique_lock<std::mutex> &lock) {
    DLOG_EVERY_N(INFO, 10000) 
        << "check switch " << this->task_switch_to_infer_
        << " " << this->num_infering_model_
        << " " << ctrl::Controller::Get()->IsInferIdle();
    return this->task_switch_to_infer_
        && this->num_infering_model_.load(std::memory_order_relaxed) == 0
        && ctrl::Controller::Get()->IsInferIdle();
  };

  auto check_switch = [this, &check_switch_unlock]() {
    std::unique_lock lock{task_switch_mutex_};
    return check_switch_unlock(lock);
  };

  auto do_switch = [this, &check_switch_unlock]() -> std::pair<bool, int> {
    // note that during the switch, infer request may come in
    std::unique_lock lock{this->task_switch_mutex_};

    int reclaim_cnt = 0;
    const int rank = 0;
    std::array<
        std::unique_lock<std::mutex>, MAX_DEVICE_NUM
    > cold_cache_locks;

    if (!check_switch_unlock(lock)) { 
      // return std::make_pair(false, -1); 
      goto cannel_switch;
    }

    LOG_IF(INFO, Config::log_task_switch) 
        << "[InferModelStore] [Task Switch] start";

    for (auto &cold_model_cache : ColdModelCache::cold_model_caches_) {
      cold_cache_locks[cold_model_cache->device_id_] = cold_model_cache->Lock();
    }

    // auto cold_cache_lock = ColdModelCache::Get()->Lock();
    for (auto &&[name, model] : this->models_) {
      if (name == "dummy") continue;
      int device_id = model->device_.device_id;
      std::unique_lock<std::mutex> model_lock{model->muts_[rank]};
      reclaim_cnt += model->ReclaimMemory(rank, cold_cache_locks[device_id], 
                                          model_lock, nullptr);

      if (!check_switch_unlock(lock)) { 
        goto cannel_switch; 
      }
    }
exit:
    return std::make_pair(true, reclaim_cnt);

cannel_switch:
    return std::make_pair(false, reclaim_cnt);
  };

  while (true) {
    if (check_switch()) {
      auto prepare_switch_time = Profiler::Get()->Now();
      std::this_thread::sleep_for(Config::task_switch_delay_ms * 1ms);
      
      auto [switch_, num_reclaim] = do_switch();
      if (!switch_) {
        LOG(INFO) << "[InferModelStore] [Task Switch] cancelled, reclaim " 
                  << num_reclaim << " models";
      } else {
        LOG(INFO) << "[InferModelStore] [Task Switch] done, reclaim " 
                  << num_reclaim << " models";
        Profiler::Get()->RecordPerf(Profiler::PerfItem::InferNumModelOnSwitch, 
                                    num_reclaim);
        ctrl::Controller::Get()->ResumeTrain();
        task_switch_to_infer_ = false;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
}


}  // namespace colserve