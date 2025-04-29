#ifndef COLSERVE_INFER_MODEL_STORE_H
#define COLSERVE_INFER_MODEL_STORE_H

#include <server/resource_manager.h>
#include <server/tvm/executor.h>
#include <server/grpc/grpc_server.h>
#include <server/schedule/job_queue.h>
#include <server/profiler.h>
#include <server/model_store/infer_model.h>
#include <server/config.h>

#include <common/util.h>

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

class InferModelStore {
 public:
  static InferModelStore* Get();
  static void Init(const std::filesystem::path &infer_store_path);
  static bool Initialized() { return Get()->initialized_; }
  static bool Shutdown();

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

  static void InferingInc(tvm::TVMGraph *graph, tvm::Executor *executor);
  static void InferingDec(tvm::TVMGraph *graph, tvm::Executor *executor);
  static int GetNumInferingModel() {
    if (Config::no_infer) {
      return 0;
    }
    return Get()->num_infering_model_.load(std::memory_order_relaxed); 
  }
  static size_t GetInferingModelNbytes() {
    if (Config::no_infer) {
      return 0;
    }
    return Get()->infering_model_nbytes_.load(std::memory_order_relaxed); 
  }

  Model* GetModel(const std::string &name);
  size_t NumJobs();

  void ClearColdCache();
  void ClearWarmCache();

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
  
  // simulate global infer request queue
  std::set<size_t> queing_infer_reqs_;

  std::unique_ptr<std::thread> monitor_thread_;

};




} // namespace colserve

#endif

