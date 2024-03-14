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

class InferModelStore {
 public:
  static InferModelStore* Get();
  static void Init(const std::filesystem::path &infer_store_path);
  static bool Shutdown() { return true; }

  static void WarmupDone();
  static void UpdateLastInferTime() {
    std::unique_lock lock{InferModelStore::Get()->mutex_};
    InferModelStore::Get()->last_infer_time_ = Profiler::Now(); 
  }
  static size_t GetModelRank() {
    return InferModelStore::Get()->model_rank_.fetch_add(1, std::memory_order_relaxed); 
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

  bool warmup_done_;
  std::atomic<size_t> model_rank_;
  std::unordered_map<std::string, std::unique_ptr<Model>> models_;

  std::mutex mutex_;
  Profiler::time_point_t last_infer_time_;
  

  std::unique_ptr<std::thread> monitor_thread_;

  // std::mutex task_switch_mutex_;
  // std::atomic<int> task_switch_control_cnter_{static_cast<int>(TaskSwitchStatus::kNotAddWorker)};
  // std::unique_ptr<std::thread> task_switch_control_;
  
  // std::atomic<int> task_switch_enter_cnt_{0};

};




} // namespace colserve

#endif

