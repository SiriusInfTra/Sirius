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
  static memory_mb_t PredictTrainMemUsageMB(int device_id, bool verbose);
  static memory_mb_t PredictTrainMemUsageMB(int device_id, int batch_size, 
                                            bool verbose);

  // TODO: adjust for re-balance training
  // TODO: guarantee either all worker batch size is greater than 0,
  //   or all worker batch size less equal to 0
  static std::vector<AdjustPlan> GetInferRequireMemAdjustPlan(
      int device_id, memory_mb_t required_mem_mb,
      memory_mb_t cold_cache_free_mem_mb);
  static std::vector<AdjustPlan> GetInferRequireMemAdjustPlanWithInLock(
      int device_id, memory_mb_t required_mem_mb, 
      memory_mb_t cold_cache_free_mem_mb,
      std::unique_lock<std::mutex> &cold_cache_lock);

  static std::vector<AdjustPlan> GetInferReleaseMemAdjustPlan();
  static std::vector<AdjustPlan> GetInferReleaseMemAdjustPlanWithInLock(
      std::vector<std::unique_lock<std::mutex>> &cold_cache_locks);

  // TODO: Refactor the interface
  static std::vector<AdjustPlan> GetLLMInferReleaseMemAdjustPlan(
      std::unique_lock<std::mutex> &kvc_pool_lock); // will be called within kv-cache-pool lock
  static std::vector<AdjustPlan> GetLLMInferRequireMemAdjustPlan(
      int device_id, memory_mb_t required_mem_mb,
      std::unique_lock<std::mutex> &kvc_pool_lock);

  static void LoadTrainInfo();
  static void ResetTrainInfo();

  TrainAdjuster();
  
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

  bool CheckImbalance(int proposed_same_batch_size);
  
  void FillAdjustPlan(const std::vector<int> &target_batch_sizes, 
                      std::vector<AdjustPlan> &plans);
  void FillSameAdjustPlan(AdjustPlan plan, 
                          std::vector<AdjustPlan> &plans);
  void FillAdjustPlanOnlyAdjustOne(int rank, AdjustPlan plan, 
                                   std::vector<AdjustPlan> &plans);

  memory_mb_t GetTrainAvailMemMBMinusColdCacheReserve(
      int device_id, bool verbose);

  // TODO: change to profile-based method
  //   to automatically modeling training memory
  std::pair<double, double> GetModelMemParam();
  std::pair<double, double> GetModelMemParam(const std::string &model_name);

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