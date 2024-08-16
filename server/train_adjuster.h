#pragma once

#include <common/util.h>

#include <atomic>
#include <array>


namespace colserve {

class TrainAdjuster {
 public: 
  struct AdjustPlan {
    int batch_size;

  };

  // we assume device_id and training rank are the same
  static void Init(); 
  static double PredictTrainMemUsageMB(int device_id, bool verbose);

  // TODO: adjust for re-balance training
  static std::vector<AdjustPlan> GetInferRequireMemAdjustPlan(
      int device_id, memory_mb_t required_mem_mb, 
      memory_mb_t cold_cache_free_mem_mb);
  static std::vector<AdjustPlan> GetInferReleaseMemAdjustPlan(int device_id);

  static void LoadTrainInfo();
  static void ResetTrainInfo();

  TrainAdjuster();
  
  // static double Get
  
 private:
  static std::unique_ptr<TrainAdjuster> adjuster_;

  friend class TrainLauncher;
  friend class ResourceManager;

  enum {
    INVALID_BATCH_SIZE = std::numeric_limits<int>::min(),
  };

  int PredictTargetBatchSize(int device_id, memory_mb_t memory_mb);
  int GetDeltaBatchSize(int device_id, memory_mb_t memory_mb);
  void UpdateCachedTrainInfoByAdjustPlan(
      const std::vector<AdjustPlan> &adjust_plans, 
      std::unique_lock<std::mutex> &lock);
  std::pair<double, double> GetModelMemParam();

  int cached_train_world_size_;
  std::array<int, MAX_DEVICE_NUM> cached_target_batch_sizes_;
  std::mutex mut_;
  
};

std::ostream& PrettyPrintAdjustPlans(
    std::ostream &os, 
    const std::vector<TrainAdjuster::AdjustPlan> &plan);

std::string PrettyPrintAdjustPlans(
    const std::vector<TrainAdjuster::AdjustPlan> &plan);


} // namespace colserve