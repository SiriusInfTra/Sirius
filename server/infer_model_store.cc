#include <common/dtype_helper.h>
#include <server/logging_as_glog.h>
#include <server/infer_model_store.h>
#include <server/infer_model.h>
#include <server/train_launcher.h>
#include <server/controller.h>
#include <server/profiler.h>
#include <server/config.h> 

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

std::unique_ptr<InferModelCache> InferModelCache::infer_model_cache_ = nullptr;
std::unique_ptr<ColdModelCache> ColdModelCache::cold_model_cache_ = nullptr;
std::unique_ptr<InferModelStore> InferModelStore::infer_model_store_ = nullptr;

std::unique_lock<std::mutex> InferModelCache::ReserveCache(
    const std::string &model_name, size_t rank) {
  if (!InferModelCache::Enable()) {
    return std::unique_lock<std::mutex>{};
  }

  std::unique_lock lock{infer_model_cache_->mut_};

  auto model = InferModelStore::Get()->GetModel(model_name);
  CHECK(infer_model_cache_ != nullptr && model != nullptr);
  infer_model_cache_->MaybeAddCacheItem(model_name, model);

  std::unique_lock reserved_lock{infer_model_cache_->warm_cache_[model_name]->mut};
  infer_model_cache_->ReserveCacheInternal(model_name, rank,
                                           reserved_lock);

  return reserved_lock;
}

std::unique_lock<std::mutex> InferModelCache::OrderedReserveCache(
    const std::string &model_name, size_t rank,
    const std::vector<std::shared_ptr<Job>> &jobs) {
  if (!InferModelCache::Enable()) {
    return std::unique_lock<std::mutex>{};
  }
  CHECK(!jobs.empty());
  auto infer_req_id = jobs[0]->GetInferData()->GetId();

  // auto _t = Profiler::Now();
  std::unique_lock lock{infer_model_cache_->mut_};
  // auto get_lock_time = Profiler::MilliFrom(_t);
  // LOG(INFO) << model_name << " get model cache lock " << get_lock_time << " ms";

  auto model = InferModelStore::Get()->GetModel(model_name);
  CHECK(infer_model_cache_ != nullptr && model != nullptr);
  infer_model_cache_->MaybeAddCacheItem(model_name, model);

  auto t0 = Profiler::Now();
  // LOG(INFO) << "try ordered reserve " << model_name << " req_id " << infer_req_id;
  infer_model_cache_->fifo_cv_.wait(lock, [&lock, infer_req_id]() {
    std::unique_lock ims_lock{InferModelStore::Get()->mutex_};
    if (InferModelStore::Get()->queing_infer_reqs_.empty()) {
      return true;
    } else {
      return *InferModelStore::Get()->queing_infer_reqs_.begin() == infer_req_id;
    }
  });

  DLOG(INFO) << "[InferModelCache] " << "ordered reserve " 
             << model_name << " req_id " << infer_req_id 
             << " wait " << Profiler::MilliFrom(t0) << " ms";

  std::unique_lock reserved_lock{infer_model_cache_->warm_cache_[model_name]->mut};

  // auto t1 = Profiler::Now();
  infer_model_cache_->ReserveCacheInternal(model_name, rank,
                                           reserved_lock);
  // LOG(INFO) << "ReserveCacheInternal " << model_name << " " << Profiler::MilliFrom(t1) << " ms";

  // auto reserved_lock = ReserveCache(model_name, rank);
  for (const auto &job : jobs) {
    std::unique_lock ims_lock{InferModelStore::infer_model_store_->mutex_};
    InferModelStore::Get()->queing_infer_reqs_.erase(job->GetInferData()->GetId());
  }

  infer_model_cache_->fifo_cv_.notify_all();
  return reserved_lock;
}

void InferModelCache::MaybeAddCacheItem(const std::string &model_name, 
                                        Model *model) {
  if (warm_cache_.count(model_name) == 0) {
    warm_cache_[model_name] = std::make_unique<CacheItem>();
    warm_cache_[model_name]->model = model;
    warm_cache_[model_name]->cached = false;
  }
}

void InferModelCache::ReserveCacheInternal(
    const std::string &model_name, size_t rank,
    std::unique_lock<std::mutex> &reserved_lock) {
  auto model = InferModelStore::Get()->GetModel(model_name);
  CHECK(model != nullptr);

  std::stringstream ss;
  ss << "[InferModelCache] " << "reserve " << model_name;

  // 1. already cached
  if (infer_model_cache_->warm_cache_[model_name]->cached) {
    ss << " already cached";
    LOG(INFO) << ss.str();
    return ;
  }

  // 2. not cached, evict if necessary
  auto nbytes = infer_model_cache_->cached_nbytes_ + model->GetMemoryNbytes(0);
  if (nbytes > Config::max_cold_cache_nbytes) {
    // std::vector<std::pair<std::string, size_t>> coldest_model{
    //   infer_model_cache_->warm_cache_[model], infer_model_cache_->hotness_.end()};
    std::vector<Model*> coldest_model;
    for (auto & it : infer_model_cache_->warm_cache_) {
      if (it.second->cached) {
        coldest_model.push_back(it.second->model);
      }
    }
    std::sort(coldest_model.begin(), coldest_model.end(), 
        [](Model *a, Model *b) { return a->GetHotness() < b->GetHotness(); });

    size_t reclaim_nbytes = 0;
    ss << " | evict";
    for (auto cm : coldest_model) {
      if (nbytes - reclaim_nbytes > Config::max_cold_cache_nbytes) {
        auto &cm_name = cm->GetName();
        std::unique_lock item_lock{infer_model_cache_->warm_cache_[cm_name]->mut};
        bool res = cm->ReclaimMemory(rank);
        if (res) {
          ss << " " << cm_name << "(hot=" << cm->GetHotness() << ")";
          infer_model_cache_->warm_cache_[cm_name]->cached = false;
          reclaim_nbytes += cm->GetMemoryNbytes(rank);
        }
      } else {
        break;
      }
    }
    nbytes -= reclaim_nbytes;
  }

  infer_model_cache_->cached_nbytes_ = nbytes;
  infer_model_cache_->warm_cache_[model_name]->cached = true;
  LOG(INFO) << ss.str() << " | now cached_nbytes=" << sta::ByteDisplay(nbytes);
}

InferModelStore* InferModelStore::Get() {
  if (infer_model_store_ == nullptr) {
    LOG(FATAL) << "InferModelStore not initialized";
  }
  return infer_model_store_.get();
}

void InferModelStore::Init(const std::filesystem::path &infer_store_path) {
  LOG(INFO) << "InferModelStore start initializing";

  InferModelCache::Init();
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
  infer_model_store_->initialized_ = true;
  LOG(INFO) << "InferModelStore initialized";
}

void InferModelStore::WarmupDone() {
  infer_model_store_->warmup_done_ = true;
  LOG(INFO) << "[InferModelStore] warmup done, num ready model "
            << Model::GetNumModel(Model::Status::kReady);
}

bool InferModelStore::AddJob(const std::string &model_name,
                             network::InferHandler::InferData* data) {
  auto model = infer_model_store_->GetModel(model_name);
  if (!model) {
    return false;
  }

  {
    std::unique_lock lock{infer_model_store_->mutex_};
    infer_model_store_->queing_infer_reqs_.insert(data->GetId());
  }
  model->AddJob(data);
  return true;
}

void InferModelStore::InferingInc(tvm::Executor *executor) {
  std::unique_lock lock{Get()->task_switch_mutex_, std::defer_lock};
  if (Config::IsSwitchMode()) {
    lock.lock();
  }
  // Get()->task_switch_cv_.wait(lock, []() {
  //   return Get()->task_switch_ctrl_.load() != static_cast<int>(TaskSwitchStatus::kReclaimInfer);
  // });
  Get()->num_infering_model_.fetch_add(1, std::memory_order_relaxed);
  Get()->infering_model_nbytes_.fetch_add(executor->GetStorageSize(), std::memory_order_relaxed); 

  // if (res == 0) {
  //   Get()->task_switch_ctrl_.store(static_cast<int>(TaskSwitchStatus::kInfering));
  // }
}

void InferModelStore::InferingDec(tvm::Executor *executor) {
  // std::unique_lock lock{Get()->task_switch_mutex_};
  // CHECK(Get()->task_switch_ctrl_.load() == static_cast<int>(TaskSwitchStatus::kInfering));
  CHECK(Get()->num_infering_model_.load(std::memory_order_relaxed) > 0);
  Get()->num_infering_model_.fetch_sub(1, std::memory_order_relaxed);
  Get()->infering_model_nbytes_.fetch_sub(executor->GetStorageSize(), std::memory_order_relaxed);
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
  auto &cold_cache = ColdModelCache::Get();
  auto cold_cache_lock = cold_cache.Lock();
  int rank = 0;
  for (auto &&[name, model]: models_) {
    auto &&[evict_groups_id, succ] = cold_cache.PopCacheItem(name, rank, cold_cache_lock);
    if (succ) { model->ClearColdCache(evict_groups_id, rank, cold_cache_lock); }
  }
  CHECK_EQ(cold_cache.GetCachedNbytes(), 0);
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

std::tuple<std::vector<size_t>, std::vector<std::pair<std::string, std::vector<size_t>>>, bool>
ColdModelCache::PushCacheItem(const std::string& name, size_t rank, std::vector<size_t> groups_nbytes, size_t total_nbytes, std::unique_lock<std::mutex> &lock) {
  DLOG(INFO) << "PushCacheItem, name = " << name << ", rank = " << rank << ", groups_nbytes = " << groups_nbytes << ", total_nbytes = " << total_nbytes;
  if (cold_cache_.count(name) != 0) { return {{}, {}, false}; }
  std::vector<std::pair<std::string, std::vector<size_t>>> evict_models;
  auto* cache_item = new CacheItem();
  cache_item->cached_group_nbytes = 0;
  cache_item->model = InferModelStore::Get()->GetModel(name);
  size_t model_max_cached_nbytes = static_cast<size_t>(total_nbytes * Config::cold_cache_ratio);;
  for (size_t k = 0; k < groups_nbytes.size(); ++k) {
    DLOG(INFO) << "k = " << k << ".";
    if ((k == 0 || cache_item->cached_group_nbytes + groups_nbytes[k] / 2 <= model_max_cached_nbytes) 
      && (cache_item->cached_group_nbytes + groups_nbytes[k] < Config::max_cold_cache_nbytes)) {
      cache_item->cached_groups_id.push_back(k);
      cache_item->cached_group_nbytes += groups_nbytes[k];
    } else {
      break;
    }
  }
  LOG(INFO) << "decide to cache group = " << cache_item->cached_groups_id << ", total" << groups_nbytes.size() << ".";

  std::vector<Model*> coldest_model;
  for (auto &&[name, cache_item] : cold_cache_) {
    coldest_model.push_back(cache_item->model);
  }

  DLOG(INFO) << "check whether should evict models.";
  // TODO: make snapshot
  std::sort(coldest_model.begin(), coldest_model.end(), 
      [](Model *a, Model *b) { return a->GetHotness() > b->GetHotness(); /* descending */ });
  while (cold_cached_nbytes_ + cache_item->cached_group_nbytes > Config::max_cold_cache_nbytes) {
    DLOG(INFO) << "should evict models.";
    auto &model_id = coldest_model.back()->GetName();
    auto &&[cached_groups_id, succ] = PopCacheItem(model_id, rank, lock);
    CHECK(succ);
    evict_models.emplace_back(model_id, std::move(cached_groups_id));
    coldest_model.pop_back();
  }
  cold_cached_nbytes_.fetch_add(cache_item->cached_group_nbytes, std::memory_order_relaxed);
  DLOG(INFO) << "put to cold_cache_.";
  CHECK(cold_cache_.emplace(std::make_pair(name, cache_item)).second == true);
  DLOG(INFO) << "cached_groups_id = " << cache_item->cached_groups_id;
  return {cache_item->cached_groups_id, evict_models, true};
}

std::pair<std::vector<size_t>, bool> ColdModelCache::PopCacheItem(const std::string& name,
    size_t rank, std::unique_lock<std::mutex> &lock) {
  auto iter = cold_cache_.find(name);
  if (iter == cold_cache_.cend()) { return {{}, false}; }
  CHECK(iter != cold_cache_.cend());
  auto cached_groups_id = iter->second->cached_groups_id;
  cold_cached_nbytes_.fetch_sub(iter->second->cached_group_nbytes, std::memory_order_relaxed);
  cold_cache_.erase(iter);
  return {cached_groups_id, true};
}

}  // namespace colserve