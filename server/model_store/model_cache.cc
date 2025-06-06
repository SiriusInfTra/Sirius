#include <server/tvm/graph.h>
#include <server/logging_as_glog.h>
#include <server/model_store/model_cache.h>
#include <server/resource_manager.h>
#include <server/model_store/infer_model_store.h>

#include <common/device_manager.h>
#include <common/util.h>
#include <algorithm>

namespace colserve {

std::vector<std::unique_ptr<WarmModelCache>> WarmModelCache::warm_model_caches_;

std::vector<std::unique_ptr<ColdModelCache>> ColdModelCache::cold_model_caches_;

ColdModelCache::ReservePolicy ColdModelCache::reserve_policy_on_release = \
    ColdModelCache::ReservePolicy::kNotReserve;

ColdModelCache::ReservePolicy ColdModelCache::reserve_policy_on_adjust = \
    ColdModelCache::ReservePolicy::kMaxCap;
  
constexpr bool warm_cache_try_evict = true;

void WarmModelCache::Init() {
  for (int i = 0; i < sta::DevMgr::GetNumVisibleGpu(); i++) {
    warm_model_caches_.push_back(std::make_unique<WarmModelCache>(i));
  }
}

WarmModelCache* WarmModelCache::Get(int device_id) {
  auto ret = warm_model_caches_.at(device_id).get();
  CHECK(ret != nullptr);
  return ret;
}

std::unique_lock<std::mutex> WarmModelCache::ReserveCache(
    const std::string &model_name, size_t rank) {
  if (!WarmModelCache::Enable()) {
    return std::unique_lock<std::mutex>{};
  }

  std::unique_lock lock{mut_};

  auto model = InferModelStore::Get()->GetModel(model_name);
  // CHECK(warm_model_caches_ != nullptr && model != nullptr);
  CHECK(model != nullptr);
  MaybeAddCacheItem(model_name, model);

  std::unique_lock reserved_lock{warm_cache_items_[model_name]->mut};
  ReserveCacheInternal(model_name, rank, reserved_lock);

  return reserved_lock;
}

std::unique_lock<std::mutex> WarmModelCache::OrderedReserveCache(
    const std::string &model_name, size_t rank,
    const std::vector<std::shared_ptr<Job>> &jobs) {
  if (!WarmModelCache::Enable()) {
    return std::unique_lock<std::mutex>{};
  }

  CHECK(!jobs.empty());
  auto infer_req_id = jobs[0]->GetInferData()->GetId();

  // auto _t = Profiler::Now();
  std::unique_lock lock{mut_};
  // auto get_lock_time = Profiler::MilliFrom(_t);
  // LOG(INFO) << model_name << " get model cache lock " << get_lock_time << " ms";

  auto model = InferModelStore::Get()->GetModel(model_name);
  // CHECK(warm_model_caches_ != nullptr && model != nullptr);
  CHECK(model != nullptr);
  MaybeAddCacheItem(model_name, model);

  auto t0 = Profiler::Now();
  // LOG(INFO) << "try ordered reserve " << model_name << " req_id " << infer_req_id;

// #if 0
//   // fifo may cause increasing latency
//
//   warm_model_caches_->fifo_cv_.wait(lock, [&lock, infer_req_id]() {
//     std::unique_lock ims_lock{InferModelStore::Get()->mutex_};
//     if (InferModelStore::Get()->queing_infer_reqs_.empty()) {
//       return true;
//     } else {
//       return *InferModelStore::Get()->queing_infer_reqs_.begin() == infer_req_id;
//     }
//   });
// #endif

  DLOG(INFO) << "[InferModelCache] " << "ordered reserve " 
             << model_name << " req_id " << infer_req_id 
             << " wait " << Profiler::MilliFrom(t0) << " ms";

  std::unique_lock reserved_lock{warm_cache_items_[model_name]->mut};

  // auto t1 = Profiler::Now();
  ReserveCacheInternal(model_name, rank,
                                           reserved_lock);
  // LOG(INFO) << "ReserveCacheInternal " << model_name << " " << Profiler::MilliFrom(t1) << " ms";

  // auto reserved_lock = ReserveCache(model_name, rank);
  for (const auto &job : jobs) {
    std::unique_lock ims_lock{InferModelStore::infer_model_store_->mutex_};
    InferModelStore::Get()->queing_infer_reqs_.erase(job->GetInferData()->GetId());
  }

  fifo_cv_.notify_all();
  return reserved_lock;
}

void WarmModelCache::MaybeAddCacheItem(const std::string &model_name, 
                                        Model *model) {
  if (warm_cache_items_.count(model_name) == 0) {
    warm_cache_items_[model_name] = std::make_unique<CacheItem>();
    warm_cache_items_[model_name]->model = model;
    warm_cache_items_[model_name]->cached = false;
  }
}

void WarmModelCache::ReserveCacheInternal(
    const std::string &model_name, size_t rank,
    std::unique_lock<std::mutex> &reserved_lock) {
  auto model = InferModelStore::Get()->GetModel(model_name);
  CHECK(model != nullptr);

  std::stringstream ss;
  ss << "[WarmModelCache] " << "reserve " << model_name;

  // 1. already cached
  if (warm_cache_items_[model_name]->cached) {
    ss << " already cached";
    LOG_IF(INFO, Config::log_warm_cache) << ss.str();
    return ;
  }

  // 2. not cached, evict if necessary
  auto nbytes = cached_nbytes_ + model->GetMemoryNbytes(0);
  if (nbytes > Config::max_warm_cache_nbytes) {
    // std::vector<std::pair<std::string, size_t>> coldest_model{
    //   infer_model_cache_->warm_cache_items_[model], infer_model_cache_->hotness_.end()};
    std::vector<Model*> coldest_model;
    for (auto & it : warm_cache_items_) {
      if (it.second->cached) {
        coldest_model.push_back(it.second->model);
      }
    }
    std::sort(coldest_model.begin(), coldest_model.end(), 
        [](Model *a, Model *b) { return a->GetHotness() < b->GetHotness(); });

    size_t reclaim_nbytes = 0;
    std::set<std::string> reclaimed_model;
    ss << " | evict";
    for (int try_cnt = 0; try_cnt < 2; try_cnt++) {
      if (!warm_cache_try_evict && try_cnt == 0) {
        continue;
      }
      for (auto cm : coldest_model) {
        if (nbytes > Config::max_warm_cache_nbytes + reclaim_nbytes) {
          if (cm == model) { continue; }
          auto &cm_name = cm->GetName();
          if (reclaimed_model.count(cm_name) > 0) { continue; }
          std::unique_lock warm_cache_lock{warm_cache_items_[cm_name]->mut, 
                                           std::defer_lock}; /* slow */
          if (try_cnt == 0) {
            warm_cache_lock.try_lock();
            if (!warm_cache_lock.owns_lock()) { continue; }
          } else {
            warm_cache_lock.lock();
          }
          bool res = cm->ReclaimMemory(rank, model);
          if (res) {
            ss << " " << cm_name << "(hot=" << cm->GetHotness() << ")";
            warm_cache_items_[cm_name]->cached = false;
            reclaim_nbytes += cm->GetMemoryNbytes(rank);
            reclaimed_model.insert(cm_name);
          }
        } else {
          break;
        }
      }
    }
    nbytes -= reclaim_nbytes;
  }

  cached_nbytes_ = nbytes;
  warm_cache_items_[model_name]->cached = true;
  LOG_IF(INFO, Config::log_warm_cache) 
      << ss.str() << " | now cached_nbytes=" << sta::PrintByte(nbytes);
}

// ==========================================================================
// ==========================================================================

void ColdModelCache::Init() {
  for (int i = 0; i < sta::DevMgr::GetNumVisibleGpu(); i++) {
    cold_model_caches_.push_back(std::make_unique<ColdModelCache>(i));
  }
}

ColdModelCache* ColdModelCache::Get(int device_id) {
  auto ret = cold_model_caches_.at(device_id).get();
  CHECK(ret != nullptr);
  return ret;
}

std::tuple<ColdModelCache::group_id_list, ColdModelCache::evict_list, bool>
ColdModelCache::PushCacheItem(
    const std::string& name, size_t rank, 
    std::vector<size_t> groups_nbytes, size_t total_nbytes, 
    std::unique_lock<std::mutex> &lock, Model *source_model) {
  LOG_IF(INFO, Config::log_cold_cache) 
      << "PushCacheItem, name = " << name 
      << ", rank = " << rank 
      << ", groups_nbytes = " << groups_nbytes 
      << ", total_nbytes = " << total_nbytes;
  if (cold_cache_items_.count(name) != 0) { return {{}, {}, false}; }

  auto* cache_item = new CacheItem();
  cache_item->cached_group_nbytes = 0;
  cache_item->model = InferModelStore::Get()->GetModel(name);
  size_t model_max_cached_nbytes = 
      Config::cold_cache_max_capacity_nbytes > 0 
      ? static_cast<size_t>(total_nbytes * Config::cold_cache_ratio) 
      : 0;
  size_t uncache_nbytes = 0;
  DLOG(INFO) << "model_max_cached_nbytes: " << model_max_cached_nbytes;

  // ----------------------------------------------
  // 1. determine the cache group, uncached nbytes
  // ----------------------------------------------
  for (size_t k = 0; k < groups_nbytes.size(); ++k) {
    memory_byte_t group_nbytes = tvm::GetMemBlockAlignedNBytes(groups_nbytes[k]);
    if (model_max_cached_nbytes > 0 
        && (k == 0 || cache_item->cached_group_nbytes + groups_nbytes[k] / 2 <= model_max_cached_nbytes)
    ) {
      cache_item->cached_groups_id.push_back(k);
      cache_item->cached_group_nbytes += group_nbytes;
    } else {
      for (; k < groups_nbytes.size(); ++k) {
        if (Config::group_param_load && Config::group_param_nbytes_with_fragment) {
          uncache_nbytes += tvm::GetMemBlockAlignedNBytes(groups_nbytes[k]);
        } else {
          uncache_nbytes += groups_nbytes[k];
        }

      }
      break;
    }
  }
  // CHECK_EQ(uncache_nbytes, 0);
  LOG_IF(INFO, Config::log_cold_cache) 
      << "[ColdModelCache] cache_item->cached_group_nbytes: " 
      << cache_item->cached_group_nbytes 
      << " uncache_nbytes: " << uncache_nbytes;
  CHECK_EQ(cache_item->cached_group_nbytes + uncache_nbytes, total_nbytes);
  LOG_IF(INFO, Config::log_cold_cache) 
      << "[ColdModelCache] decide to cache " << name 
      << " decide to cache group = [ " << cache_item->cached_groups_id << " ]," 
      << " total " << groups_nbytes.size() << ".";

  std::vector<Model*> coldest_model;
  for (auto &&[name, cache_item] : cold_cache_items_) {
    coldest_model.push_back(cache_item->model);
  }

  DLOG_IF(INFO, Config::log_cold_cache) << "check whether should evict models.";
  evict_list evict_models;
  // CHECK_EQ(uncache_nbytes, 0);
  
  memory_byte_t before_capacity_nbytes = current_capacity_nbytes_;
  memory_byte_t before_cache_nbytes = current_cached_nbytes_;
  memory_byte_t before_infer_nbytes = 
      ResourceManager::GetInferMemoryMB(sta::DevMgr::GetCurrentDevice()) * 1_MB;
  memory_byte_t before_ic_nbytes = before_infer_nbytes 
                                   - current_cached_nbytes_ 
                                   + current_capacity_nbytes_;
  memory_byte_t evict_nbytes = 0;

  // -------------------
  // 2. cache the group
  // -------------------
  if (current_capacity_nbytes_ + total_nbytes <= Config::cold_cache_max_capacity_nbytes
  ) { /* 2.1 resereve in cache */ 
    // note that all memory (whether or not model is partial cached) is reserved in cache
    current_capacity_nbytes_ += total_nbytes; 
    current_cached_nbytes_ += cache_item->cached_group_nbytes;
    DLOG_IF(INFO, Config::log_cold_cache) 
        << "[ColdModelCache] Cache Succes, update Capacity=" 
        << sta::PrintByte(current_capacity_nbytes_);
  } else { /* 2.2 reach limit, evict the coldest */  
    // CHECK_GE((Config::cold_cache_max_capacity_nbytes + Config::cold_cache_min_capacity_nbytes) / 2, 
    //          cache_item->cached_group_nbytes);
    memory_byte_t evict_to_nbytes = (Config::cold_cache_max_capacity_nbytes 
                                     + Config::cold_cache_min_capacity_nbytes) / 2;
    CHECK_GE(evict_to_nbytes, cache_item->cached_group_nbytes);

    memory_byte_t old_capacity_nbytes = current_capacity_nbytes_;
    memory_byte_t old_cached_nbytes = current_cached_nbytes_;
    evict_models = GetEvictModels(evict_to_nbytes , 
                                  {cache_item->model, source_model}, lock);
    evict_nbytes = old_cached_nbytes - current_cached_nbytes_;
    current_capacity_nbytes_ = current_cached_nbytes_;
    current_capacity_nbytes_ += cache_item->cached_group_nbytes;
    current_cached_nbytes_ += cache_item->cached_group_nbytes;
    DLOG_IF(INFO, Config::log_cold_cache) 
        << "[PushCacheItem] Shrink Cache, " 
        << "capacity: " << sta::PrintByte(old_capacity_nbytes) 
        << " -> " << sta::PrintByte(current_capacity_nbytes_) << ", "
        << "cached: " << sta::PrintByte(old_cached_nbytes) 
        << " -> " << sta::PrintByte(current_cached_nbytes_) << ", "
        << "evict_to: " << sta::PrintByte(evict_to_nbytes) << ".";
  }
  CHECK(cold_cache_items_.emplace(std::make_pair(name, cache_item)).second == true) << name;
  DLOG_IF(INFO, Config::log_cold_cache) 
      << "cached_groups_id = " << cache_item->cached_groups_id; 

  memory_byte_t after_capacity_nbytes = current_capacity_nbytes_;
  memory_byte_t after_cache_nbytes = current_cached_nbytes_;
  memory_byte_t after_infer_nbytes = 
      ResourceManager::GetInferMemoryMB(sta::DevMgr::GetCurrentDevice()) * 1_MB;
  memory_byte_t after_ic_nbytes = after_infer_nbytes 
                                  - current_cached_nbytes_ 
                                  + current_capacity_nbytes_ 
                                  - evict_nbytes;
  LOG_IF(INFO, Config::log_cold_cache) << "[PutCacheItem] name " << name << ", " 
      << "capacity_nbytes: " << sta::PrintByte(before_capacity_nbytes) 
      << " -> " << sta::PrintByte(after_capacity_nbytes) << ", "
      << "cache_nbytes: " << sta::PrintByte(before_cache_nbytes) 
      << " -> " << sta::PrintByte(after_cache_nbytes) << ", "
      << "infer_nbytes: " << sta::PrintByte(before_infer_nbytes) 
      << " -> " << sta::PrintByte(after_infer_nbytes) << ", "
      << "ic_nbytes: " << sta::PrintByte(before_ic_nbytes) 
      << " -> " << sta::PrintByte(after_ic_nbytes) << ", "
      << "evict_nbytes: " << sta::PrintByte(evict_nbytes) << ".";

  return {cache_item->cached_groups_id, evict_models, true};
}

std::pair<std::vector<size_t>, bool> ColdModelCache::PopCacheItem(const std::string& name,
    size_t rank, bool pop_to_inference, std::unique_lock<std::mutex> &lock) {
  auto iter = cold_cache_items_.find(name);
  if (iter == cold_cache_items_.cend()) { return {{}, false}; }
  CHECK(iter != cold_cache_items_.cend());
  auto cached_groups_id = iter->second->cached_groups_id;
  current_cached_nbytes_ -= iter->second->cached_group_nbytes;
  if (pop_to_inference) {
    current_capacity_nbytes_ -= iter->second->cached_group_nbytes;
  }
  // CHECK_GE(current_capacity_nbytes_, Config::cold_cache_min_capacity_nbytes - 500_MB);
  // CHECK_LE(current_capacity_nbytes_, Config::cold_cache_max_capacity_nbytes);
  CHECK_LE(current_cached_nbytes_, current_capacity_nbytes_);
  cold_cache_items_.erase(iter);
  return {cached_groups_id, true};
}

ColdModelCache::evict_list ColdModelCache::GetEvictModels(
    memory_byte_t capacity, 
    std::array<Model*, 2> ignore_models, 
    std::unique_lock<std::mutex>& lock) {
  const size_t default_rank = 0;
  std::vector<Model*> coldest_model;
  for (auto&& [name, cache_item] : cold_cache_items_) {
    if (cache_item->model == ignore_models[0] 
       || cache_item->model == ignore_models[1]) {
      continue;
    }
    coldest_model.push_back(cache_item->model);
  }
  std::sort(coldest_model.begin(), coldest_model.end(), [](Model* a, Model* b) {
    return a->GetHotness() > b->GetHotness(); /* descending */
  });

  evict_list evict_models;
  while (current_cached_nbytes_ > capacity && !coldest_model.empty()) {
    DLOG(INFO) << "should evict models.";
    auto *model = coldest_model.back();
    auto& model_id = model->GetName();
    auto&& [cached_groups_id, succ] = PopCacheItem(
        model_id, default_rank, false, lock);
    CHECK(succ);
    evict_models.emplace_back(model_id, std::move(cached_groups_id));

    coldest_model.pop_back();
  }

  CHECK_LE(current_cached_nbytes_, capacity) 
      << "Unable to evict models to make sure current_cached_nbytes_ > capacity";
  return evict_models;
}

double ColdModelCache::GetBufferMBUnsafe() {
  auto buffer_mb = ResourceManager::GetFreeMemoryMB(device_id_, false);
  auto cold_cache_nbytes = current_cached_nbytes_;
  if (cold_cache_nbytes > Config::cold_cache_min_capacity_nbytes) {
    buffer_mb += sta::ByteToMB(
      cold_cache_nbytes - Config::cold_cache_min_capacity_nbytes);
  }
  double cur_max_buffer_mb = 
      sta::ByteToMB(Config::cold_cache_max_capacity_nbytes
                    - std::min(Config::cold_cache_min_capacity_nbytes, 
                               current_cached_nbytes_));
  buffer_mb = std::min(buffer_mb, cur_max_buffer_mb);
  return std::max(0.0, buffer_mb);
}

double ColdModelCache::GetCacheSizeMBUnsafe() {
  return sta::ByteToMB(current_capacity_nbytes_);
  // auto cold_cache_nbytes = current_cached_nbytes_;
  // auto free_memory_mb = ResourceManager::GetFreeMemoryMB(device_id_, false);
  // LOG(INFO) << "cold_cache_nbytes = " << cold_cache_nbytes << ", free_memory_mb = " << free_memory_mb;
  // return std::min(sta::ByteToMB(Config::cold_cache_max_capacity_nbytes),
  //                 sta::ByteToMB(cold_cache_nbytes) + std::max(0.0, free_memory_mb));
}

double ColdModelCache::GetReleaseReserveMemoryMBUnsafe() {
  double cached_MB = sta::ByteToMB(current_cached_nbytes_);
  double min_cap_MB = sta::ByteToMB(Config::cold_cache_min_capacity_nbytes);
  double max_cap_MB = sta::ByteToMB(Config::cold_cache_max_capacity_nbytes);
  double cap_MB = sta::ByteToMB(current_capacity_nbytes_);
  return std::max(0.0, cap_MB - cached_MB);
  // if (ColdModelCache::reserve_policy_on_release == ReservePolicy::kMaxCap) {
  //   return std::max(0.0, min_cap_MB - cached_MB);
  // } else if (ColdModelCache::reserve_policy_on_release == ReservePolicy::kMaxMinDiff) {
  //   return cached_MB > min_cap_MB 
  //          ? std::max(0.0, max_cap_MB - cached_MB) 
  //          : max_cap_MB - min_cap_MB;
  // } else {
  //   return 0.0;
  // }
}

double ColdModelCache::GetAdjustReserveMemoryMBUnsafe() {
  double cached_MB = sta::ByteToMB(current_cached_nbytes_);
  double min_cap_MB = sta::ByteToMB(Config::cold_cache_min_capacity_nbytes);
  double max_cap_MB = sta::ByteToMB(Config::cold_cache_max_capacity_nbytes);
  double cap_MB = sta::ByteToMB(current_capacity_nbytes_);
  return std::max(0.0, cap_MB - cached_MB);

  // if (ColdModelCache::reserve_policy_on_adjust == ReservePolicy::kMaxCap) {
  //   return std::max(0.0, max_cap_MB - cached_MB);
  // } else if (ColdModelCache::reserve_policy_on_adjust == ReservePolicy::kMaxMinDiff) {
  //   return cached_MB > min_cap_MB 
  //          ? std::max(0.0, max_cap_MB - cached_MB) 
  //          : max_cap_MB - min_cap_MB;
  // } else {
  //   return 0.0;
  // }
}

double ColdModelCache::GetReleaseReserveMemoryMB(
    std::unique_lock<std::mutex> &lock) {
  return GetReleaseReserveMemoryMBUnsafe();
}

double ColdModelCache::GetAdjustReserveMemoryMB(
    std::unique_lock<std::mutex> &lock) {
  return GetAdjustReserveMemoryMBUnsafe();
}

void ColdModelCache::SetNewCapacity(memory_byte_t new_capacity,
                                    std::unique_lock<std::mutex> &lock) {
  // CHECK_LE(new_capacity, Config::cold_cache_max_capacity_nbytes);
  // CHECK_GE(new_capacity, Config::cold_cache_min_capacity_nbytes);
  CHECK_GE(new_capacity, current_cached_nbytes_);
  current_capacity_nbytes_ = new_capacity;
  DLOG_IF(INFO, Config::log_cold_cache) 
      << "SetNewCapacity " << sta::PrintByte(new_capacity);
}

bool ColdModelCache::TakeSpace(memory_byte_t nbytes) {
  auto current_capacity_nbytes = current_capacity_nbytes_;
  if (current_cached_nbytes_ + nbytes <= current_capacity_nbytes_) {
    current_capacity_nbytes_ -= nbytes;
    DLOG_IF(INFO, Config::log_cold_cache) 
        << str(boost::format(
          "[ColdModelCache] TakeSpace Succ %s, current cache %s: %s -> %s, FREE: %s.")
            % sta::PrintByte(nbytes)
            % sta::PrintByte(current_cached_nbytes_)
            % sta::PrintByte(current_capacity_nbytes)
            % sta::PrintByte(current_capacity_nbytes_)
            % ResourceManager::GetFreeMemoryMB(
                sta::DevMgr::GetCurrentDevice(), false));
    return true;
  } else {
    DLOG_IF(INFO, Config::log_cold_cache) 
        << str(boost::format(
          "[ColdModelCache] TakeSpace Fail %s, current cache %s: %s -> %s, FREE: %s.")
              % sta::PrintByte(nbytes)
              % sta::PrintByte(current_cached_nbytes_)
              % sta::PrintByte(current_capacity_nbytes)
              % sta::PrintByte(current_capacity_nbytes_)
              % ResourceManager::GetFreeMemoryMB(
                  sta::DevMgr::GetCurrentDevice(), false));
    return false;
  }
}

std::string ColdModelCache::PrintCacheInfo(std::unique_lock<std::mutex> &lock) {
  return str (boost::format("Capacity: %s, Cached: %s") 
              % sta::PrintByte(GetCacheCapacity(lock)) 
              % sta::PrintByte(GetCachedNbytes(lock)));
}

} // namespace colserve