#ifndef COLSERVE_INFER_MODEL_STORE_H
#define COLSERVE_INFER_MODEL_STORE_H

#include "common/util.h"
#include "logging_as_glog.h"
#include "resource_manager.h"
#include <server/tvm/executor.h>
#include <server/grpc/grpc_server.h>
#include <server/job_queue.h>
#include <server/profiler.h>
#include <server/infer_model.h>
#include <server/config.h>

#include <atomic>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <condition_variable>
#include <utility>

namespace colserve {

class Model;
class InferModelStore;

/* [Note: Cache]
 *   we will have two level cache,
 *   1. warm cache: clasic model cache in infer server, 
 *      swap in/out model from/to the host memory to 
 *      over-subscribe the GPU memory
 * 
 *   2. cold cache: cache idle model in gpu to avoid some
 *      cold start
 */
class WarmModelCache {
 public:
  WarmModelCache() : cached_nbytes_(0) {};
  static std::unique_lock<std::mutex> ReserveCache(const std::string &name, size_t rank);
  static std::unique_lock<std::mutex> OrderedReserveCache(
      const std::string &name, size_t rank,
      const std::vector<std::shared_ptr<Job>> &jobs);
  
  static bool Enable() { return Config::max_warm_cache_nbytes != 0; }

  friend class InferModelStore;

 private:
  struct CacheItem {
    Model *model;
    bool cached;
    std::mutex mut;
  };

  static void Init() {
    infer_model_cache_ = std::make_unique<WarmModelCache>();
  }
  void MaybeAddCacheItem(const std::string &name, Model *model);
  void ReserveCacheInternal(const std::string &name, size_t rank,
                            std::unique_lock<std::mutex> &reserved_lock);

  static std::unique_ptr<WarmModelCache> infer_model_cache_;

  std::mutex mut_;
  std::condition_variable fifo_cv_;
  size_t cached_nbytes_;
  std::unordered_map<std::string, std::unique_ptr<CacheItem>> warm_cache_;

};

class ColdModelCache {
 private:
  size_t current_cached_nbytes_;
  std::mutex mut_;
  struct CacheItem {
    Model *model;
    std::vector<size_t> cached_groups_id;
    size_t cached_group_nbytes;
  };
  static std::unique_ptr<ColdModelCache> cold_model_cache_;
  std::unordered_map<std::string, std::unique_ptr<CacheItem>> cold_cache_;

 public:
  using evict_list = std::vector<std::pair<std::string, std::vector<size_t>>>;
  using group_id_list = std::vector<size_t>;
  ColdModelCache(): current_cached_nbytes_(0) {}

  friend class InferModelStore;

  /**
   * Pushes a cache item into the model store.
   *
   * @param name The name of the cache item.
   * @param rank The rank of the cache item.
   * @param groups_nbytes The vector of group sizes in bytes.
   * @param total_nbytes The total size of the cache item in bytes.
   * @param lock The unique lock used to synchronize access to the infer model store.
   * @return A tuple containing the vector of group sizes, a vector of pairs representing the names and group sizes,
   *         and a boolean indicating the success of the operation.
   */
  std::tuple<std::vector<size_t>, std::vector<std::pair<std::string, std::vector<size_t>>>, bool>
  PushCacheItem(const std::string& name, size_t rank, std::vector<size_t> groups_nbytes, size_t total_nbytes, std::unique_lock<std::mutex> &lock);


  /**
   * @brief Pops a cache item from the infer model store.
   *
   * This function removes and returns a cache item identified by the given name and rank from the infer model store.
   *
   * @param name The name of the cache item to pop.
   * @param rank The rank of the cache item to pop.
   * @param lock A unique lock that ensures thread safety during the operation.
   * @return A pair containing the vector of sizes associated with the cache item and a boolean indicating whether the pop operation was successful.
   */
  std::pair<std::vector<size_t>, bool> PopCacheItem(const std::string& name, size_t rank, std::unique_lock<std::mutex> &lock);

  /**
   * Retrieves the list of models that are eligible for eviction from the infer model store.
   * 
   * @param capacity The desired capacity after eviction.
   * @param ignore_model 
   * @param lock A unique lock on the mutex.
   * @return The list of models that should be evicted.
   */
  evict_list GetEvictModels(long capacity, const std::string &ignore_model_name, std::unique_lock<std::mutex>& lock);


  std::unique_lock<std::mutex> Lock() {
    return std::unique_lock{mut_};
  }

  inline size_t GetCachedNbytes(std::unique_lock<std::mutex> &lock) {
    return current_cached_nbytes_;
  }


  inline double GetColdCacheFreeMemoryMB(double free_memory_MB, std::unique_lock<std::mutex> &lock) {
    if (current_cached_nbytes_ > Config::cold_cache_min_capability_nbytes){
      free_memory_MB += sta::ByteToMB(current_cached_nbytes_ - Config::cold_cache_min_capability_nbytes);
    }
    LOG(INFO) << "[ColdModelCache] FreeMemory " << free_memory_MB << "MB";
    return free_memory_MB;
  }
  

  static void Init() {
    cold_model_cache_ = std::make_unique<ColdModelCache>();
  }

  static ColdModelCache &Get() {
    CHECK(cold_model_cache_ != nullptr);
    return *cold_model_cache_;
  }


};


class InferAllocProxy {
 public:
  InferAllocProxy() {}
  static InferAllocProxy &Get() {
    return *batch_adjust_proxy_;
  }

  static void Init() {
    batch_adjust_proxy_ = std::make_unique<InferAllocProxy>();
  }

  void Lock() {

  }

  void AllocInfer(tvm::Executor &executor) {
    std::unique_lock lock{mutex_};
    CHECK_LE(executor.GetMissingStorageSizeAlign(), Config::infer_alloc_buffer_nbytes);
    if (executor.GetMissingStorageSizeAlign() > proxy_nbytes_) {
      size_t alloc_nbytes = Config::infer_alloc_buffer_nbytes - proxy_nbytes_ + executor.GetMissingStorageSizeAlign();
      proxy_nbytes_ = Config::infer_alloc_buffer_nbytes;
    }
    if (cold_cache_buffer_nbytes_ > proxy_nbytes_) {
      size_t evict_cold_cache_nbytes = cold_cache_buffer_nbytes_ - proxy_nbytes_;
    }
  }

  bool AllocCache(size_t nbytes) {
    std::unique_lock lock{mutex_};
    CHECK_LE(nbytes, Config::infer_alloc_buffer_nbytes);
    if (cold_cache_buffer_nbytes_ + nbytes > proxy_nbytes_) {
      return false;
    }
    cold_cache_buffer_nbytes_ += nbytes;
    return true;
  }

  void FreeCache(size_t nbytes) {
    std::unique_lock lock{mutex_};
    CHECK_LE(nbytes, cold_cache_buffer_nbytes_);
    cold_cache_buffer_nbytes_ -= nbytes;
  }


 private:
  size_t cold_cache_buffer_nbytes_;
  size_t proxy_nbytes_;
  std::mutex mutex_;
  static std::unique_ptr<InferAllocProxy> batch_adjust_proxy_;
};


class InferModelStore {
 public:
  static InferModelStore* Get();
  static void Init(const std::filesystem::path &infer_store_path);
  static bool Initialized() { return Get()->initialized_; }
  static bool Shutdown() { return true; }

  static void WarmupDone();
  static bool AddJob(const std::string &model_name, 
                     network::InferHandler::InferData* data);
  
  static void UpdateLastInferTime() {
    std::unique_lock lock{Get()->mutex_};
    Get()->last_infer_time_ = Profiler::Now(); 
  }
  static size_t GetModelRank() {
    return Get()->model_rank_.fetch_add(1, std::memory_order_relaxed); 
  }

  static void InferingInc(tvm::Executor *executor);
  static void InferingDec(tvm::Executor *executor);
  static int GetNumInferingModel() { 
    return Get()->num_infering_model_.load(std::memory_order_relaxed); 
  }
  static size_t GetInferingModelNbytes() { 
    return Get()->infering_model_nbytes_.load(std::memory_order_relaxed); 
  }

  Model* GetModel(const std::string &name);
  size_t NumJobs();

  void ClearColdCache();

  // void TaskSwitchEnter() { task_switch_enter_cnt_++; }
  // void TaskSwitchExit() { task_switch_enter_cnt_--; }
  // std::mutex &TaskSwitchMutex() { return task_switch_mutex_; }
  // const std::atomic<int> &TaskSwitchControlCnter() { return task_switch_control_cnter_; }
  // std::atomic<int> &MutableTaskSwitchControlCnter() { return task_switch_control_cnter_; }

  // std::condition_variable task_switch_cv;
  // pthread_barrier_t task_switch_barrier;

  // enum class TaskSwitchStatus {
  //   kExit = 0,
  //   kCancelExit = 1,
  //   kPrepareExit = 2, 
    
  //   kNotAddWorker = 3,
  //   kAddWorker = 4,
  // };

  enum class TaskSwitchStatus {
    kNotInfering = 0,
    kInfering = 1,
    kReclaimInfer = 2, 
  };

  friend class WarmModelCache;

 private:
  void ColocateMonitor();
  void TaskSwitchMonitor();

  inline double GetMaxIdleMill() { 
    if (!warmup_done_) {
      return 3000; // a default dummy value
    }
    return Config::infer_model_max_idle_ms; 
  }

  static std::unique_ptr<InferModelStore> infer_model_store_;

  bool initialized_;
  bool warmup_done_;
  std::atomic<size_t> model_rank_;
  std::unordered_map<std::string, std::unique_ptr<Model>> models_;

  std::mutex mutex_;
  Profiler::time_point_t last_infer_time_;

  std::atomic<size_t> infering_model_nbytes_{0};
  std::atomic<int> num_infering_model_{0};

  std::mutex task_switch_mutex_;
  bool task_switch_to_infer_{false};
  // std::atomic<int> task_switch_ctrl_{static_cast<int>(TaskSwitchStatus::kNotInfering)};
  // std::condition_variable task_switch_cv_;
  
  // simulate global infer request queue
  std::set<size_t> queing_infer_reqs_;

  std::unique_ptr<std::thread> monitor_thread_;
  
  // std::atomic<int> task_switch_control_cnter_{static_cast<int>(TaskSwitchStatus::kNotAddWorker)};
  // std::unique_ptr<std::thread> task_switch_control_;
  
  // std::atomic<int> task_switch_enter_cnt_{0};
};




} // namespace colserve

#endif

