#ifndef COLSERVE_INFER_MODEL_STORE_H
#define COLSERVE_INFER_MODEL_STORE_H

#include "logging_as_glog.h"
#include <server/tvm/executor.h>
#include <server/grpc/grpc_server.h>
#include <server/job_queue.h>
#include <server/profiler.h>
#include <server/infer_model.h>
#include <server/config.h>

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <condition_variable>

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
class InferModelCache {
 public:
  InferModelCache() : cached_nbytes_(0) {};
  static std::unique_lock<std::mutex> ReserveCache(const std::string &name, size_t rank);
  static std::unique_lock<std::mutex> OrderedReserveCache(
      const std::string &name, size_t rank,
      const std::vector<std::shared_ptr<Job>> &jobs);
  
  static bool Enable() { return Config::max_cache_nbytes != 0; }

  friend class InferModelStore;

 private:
  struct CacheItem {
    Model *model;
    bool cached;
    std::mutex mut;
  };

  static void Init() {
    infer_model_cache_ = std::make_unique<InferModelCache>();
  }
  void MaybeAddCacheItem(const std::string &name, Model *model);
  void ReserveCacheInternal(const std::string &name, size_t rank,
                            std::unique_lock<std::mutex> &reserved_lock);

  static std::unique_ptr<InferModelCache> infer_model_cache_;

  std::mutex mut_;
  std::condition_variable fifo_cv_;
  size_t cached_nbytes_;
  
  std::unordered_map<std::string, std::unique_ptr<CacheItem>> warm_cache_;
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

  friend class InferModelCache;

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

