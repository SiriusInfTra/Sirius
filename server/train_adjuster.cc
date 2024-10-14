#include <server/logging_as_glog.h>
#include <server/config.h>
#include <server/train_adjuster.h>
#include <server/model_store/model_cache.h>
#include <server/resource_manager.h>

#include <common/inf_tra_comm/communicator.h>

#include <boost/range/irange.hpp>
#include <math.h>


namespace colserve {

std::unique_ptr<TrainAdjuster> TrainAdjuster::adjuster_ = nullptr;

void TrainAdjuster::Init() {
  CHECK(adjuster_ == nullptr);
  adjuster_ = std::make_unique<TrainAdjuster>();
}

TrainAdjuster::TrainAdjuster() {
  cached_train_world_size_ = 0;
  for (auto &bs : cached_target_batch_sizes_) {
    bs = INVALID_BATCH_SIZE;
  }
}

void TrainAdjuster::LoadTrainInfo() {
  CHECK(adjuster_ != nullptr);

  std::unique_lock lock{adjuster_->mut_};
  auto inftra_info = ctrl::InfTraCommunicator::GetSinfo();
  adjuster_->cached_train_world_size_ = 
      ctrl::InfTraCommunicator::GetTrainWorldSize();
  for (auto rank : boost::irange(adjuster_->cached_train_world_size_)) {
    adjuster_->cached_target_batch_sizes_[rank] =
        inftra_info->GetTrainInfoUnsafe(rank)->target_batch_size;
  }
}

void TrainAdjuster::ResetTrainInfo() {
  CHECK(adjuster_ != nullptr);

  std::unique_lock lock{adjuster_->mut_};
  adjuster_->cached_train_world_size_ = 0;
  for (auto &bs : adjuster_->cached_target_batch_sizes_) {
    bs = INVALID_BATCH_SIZE;
  }
}

memory_mb_t TrainAdjuster::PredictTrainMemUsageMB(int device_id, bool verbose) {
  CHECK(adjuster_ != nullptr);
  if (!ctrl::InfTraCommunicator::GetSinfo()->IsTrainInfoValid(ctrl::kTraRank_0)) {
    return 0;
  }
  auto target_batch_size_ = 
      adjuster_->cached_target_batch_sizes_[device_id];
  return PredictTrainMemUsageMB(device_id, target_batch_size_, verbose);
}

memory_mb_t TrainAdjuster::PredictTrainMemUsageMB(
      int device_id, int batch_size, bool verbose) {
  CHECK(adjuster_ != nullptr);
  auto [base, slope] = adjuster_->GetModelMemParam();
  if (batch_size <= 0) {
    return base;
  } else {
    auto res = base + slope * batch_size;
    LOG_IF(INFO, verbose && Config::log_memory_adjust) 
        << "Predict train memory " << res
        << ", target batch size " << batch_size;
    return res;
  }
}

std::vector<TrainAdjuster::AdjustPlan>
TrainAdjuster::GetTrainRequireMemAdjustPlan(
    int device_id, memory_mb_t required_mem_mb,
    memory_mb_t cold_cache_free_mem_mb) {
  CHECK(adjuster_ != nullptr);
  std::unique_lock<std::mutex> cold_cache_lock = 
      ColdModelCache::Get(device_id)->Lock();
  return GetInferRequireMemAdjustPlanWithInLock(
      device_id, required_mem_mb, 
      cold_cache_free_mem_mb, cold_cache_lock);
}

std::vector<TrainAdjuster::AdjustPlan> 
TrainAdjuster::GetInferRequireMemAdjustPlanWithInLock(
    int device_id, 
    memory_mb_t required_mem_mb,
    memory_mb_t cold_cache_free_mem_mb,
    std::unique_lock<std::mutex> &cold_cache_lock) {
  CHECK(adjuster_ != nullptr);
  CHECK_GT(required_mem_mb, cold_cache_free_mem_mb)
      << "memory adjust is not necessary";

  std::unique_lock adjuster_lock{adjuster_->mut_};

  auto train_world_size = adjuster_->cached_train_world_size_;
  int cur_train_target_bs = 
      adjuster_->cached_target_batch_sizes_[device_id];

  if (cur_train_target_bs <= 0) {
    LOG_IF(INFO, Config::log_memory_adjust) 
        << "[InferRequireMemAdjust] target batch batch is already 0, skip adjust";
    return {};
  }

  memory_mb_t adjust_reserve_mb =
      ColdModelCache::Get(device_id)->GetAdjustReserveMemoryMBUnsafe();
  memory_mb_t adjust_batch_buf_mb = required_mem_mb 
                                    - std::max(0.0, cold_cache_free_mem_mb) 
                                    + adjust_reserve_mb;

  if (adjust_batch_buf_mb <= 0) {
    LOG_IF(INFO, Config::log_memory_adjust) 
        << "[InferRequireMemAdjust] skip adjust memory " 
        << adjust_batch_buf_mb;
    return {};
  }

  int delta_batch_size = 
      adjuster_->GetDeltaBatchSize(device_id, adjust_batch_buf_mb);
  CHECK_GE(adjust_batch_buf_mb, 0);
  int target_batch_size = cur_train_target_bs - delta_batch_size;

  // [Note: adjust plan for imbalance]
  // case 1: all batch size is same 
  //   - adjust all training if not deviate from available memory too much
  //   - otherwise, only adjust the specific device
  // case 2: has different batch size
  //   - adjust the specific device only
  //   - but check if it can be adjusted to the same batch size

  bool all_same = true;
  for (auto rank : boost::irange(train_world_size)) {
    if (adjuster_->cached_target_batch_sizes_[rank] != cur_train_target_bs) {
      all_same = false;
    }
  }

  std::vector<TrainAdjuster::AdjustPlan> adjust_plan(train_world_size);
  if (all_same) {
    bool imbalance = 
        target_batch_size > 0 /* avoid partial training workers executing */
        && adjuster_->CheckImbalance(target_batch_size);
    if (imbalance) {
      adjuster_->FillAdjustPlanOnlyAdjustOne(
        device_id, AdjustPlan{.batch_size = target_batch_size}, adjust_plan);
    } else {
      adjuster_->FillSameAdjustPlan(
        AdjustPlan{.batch_size = target_batch_size}, adjust_plan);
    }
  } else {
    int min_batch_size = std::numeric_limits<int>::max();
    for (auto rank : boost::irange(train_world_size)) {
      min_batch_size = std::min(min_batch_size, 
          rank == device_id 
          ? target_batch_size 
          : adjuster_->cached_target_batch_sizes_[rank]);
    }

    bool imbalance = 
        min_batch_size > 0 /* avoid partial training workers executing */
        && adjuster_->CheckImbalance(min_batch_size);
    if (imbalance) {
      adjuster_->FillAdjustPlanOnlyAdjustOne(
        device_id, AdjustPlan{.batch_size = min_batch_size}, adjust_plan);
    } else {
      adjuster_->FillSameAdjustPlan(
        AdjustPlan{.batch_size = target_batch_size}, adjust_plan);
    }
  }
  
  adjuster_->UpdateCachedTrainInfoByAdjustPlan(
      adjust_plan, adjuster_lock);
  adjuster_lock.unlock();

  if (Config::log_memory_adjust) {
    std::stringstream ss; 
    ss << "[InferRequireMemAdjust]"
      << " cur_train_target_bs " << cur_train_target_bs
      << " cold cache free memory " << cold_cache_free_mem_mb
      << " adjust_reserve_mb " << adjust_reserve_mb
      << " required_mem_mb " << required_mem_mb
      << " adjust_batch_buf_mb " << adjust_batch_buf_mb
      << " delta_batch_size " << delta_batch_size
      << " | require adjust plan: " << PrettyPrintAdjustPlans(adjust_plan);
    LOG(INFO) << ss.str();
  }

  return adjust_plan;
}

std::vector<TrainAdjuster::AdjustPlan>
TrainAdjuster::GetInferReleaseMemAdjustPlan() {
  CHECK(adjuster_ != nullptr);
  std::vector<std::unique_lock<std::mutex>> cold_cache_locks;
  for (auto device_id : boost::irange(sta::DeviceManager::GetNumVisibleGpu())) {
    cold_cache_locks.emplace_back(
        ColdModelCache::Get(device_id)->Lock());
  }
  return GetInferReleaseMemAdjustPlanWithInLock(cold_cache_locks);
}

std::vector<TrainAdjuster::AdjustPlan>
TrainAdjuster::GetInferReleaseMemAdjustPlanWithInLock(
    std::vector<std::unique_lock<std::mutex>> &cold_cache_locks) {
  CHECK(adjuster_ != nullptr);
  CHECK(cold_cache_locks.size() == sta::DeviceManager::GetNumVisibleGpu());
  
  std::unique_lock adjuster_lock{adjuster_->mut_};

  int train_world_size = adjuster_->cached_train_world_size_;
  // int cur_train_target_bs = 
  //     adjuster_->cached_target_batch_sizes_[0];

  std::vector<memory_mb_t> train_avail_mem_mbs,
                           free_mem_mbs,
                           reserve_mem_mbs;
  std::vector<int> cur_train_target_batches;

  for (auto device_id : boost::irange(sta::DeviceManager::GetNumVisibleGpu())) {
    train_avail_mem_mbs.push_back(
        std::max(ResourceManager::GetTrainAvailMemoryMB(device_id, false), 0.0));
    free_mem_mbs.push_back(
        std::max(ResourceManager::GetFreeMemoryMB(device_id, false), 0.0));
    reserve_mem_mbs.push_back(
        ColdModelCache::Get(device_id)->GetReleaseReserveMemoryMBUnsafe());
    cur_train_target_batches.push_back(
        adjuster_->cached_target_batch_sizes_[device_id]);
  }

  std::vector<int> target_bs_predict_by_avail_mems(train_world_size);
  std::vector<int> target_bs_calcu_by_delta_mems(train_world_size);
  std::vector<int> target_batch_sizes(train_world_size);
  for (auto rank : boost::irange(train_world_size)) {
    auto & target_bs_predict_by_avail_mem = 
        target_bs_predict_by_avail_mems[rank];
    auto & target_bs_calcu_by_delta_mem = 
        target_bs_calcu_by_delta_mems[rank];
    auto & target_batch_size = target_batch_sizes[rank];

    auto train_avail_mem_mb = train_avail_mem_mbs[rank];
    auto free_mem_mb = free_mem_mbs[rank];
    auto reserve_mem_mb = reserve_mem_mbs[rank];

    int cur_train_target_bs = cur_train_target_batches[rank];

    if (Config::use_shared_tensor) {
      target_bs_predict_by_avail_mem = adjuster_->PredictTargetBatchSize(
          0, std::max(train_avail_mem_mb - reserve_mem_mb, 0.0));

      auto release_mem_mb = std::max(free_mem_mb - reserve_mem_mb, 0.0);
      target_bs_calcu_by_delta_mem =
          cur_train_target_bs + adjuster_->GetDeltaBatchSize(0, release_mem_mb);

      // ensure not OOM but also not decrease batch size
      target_batch_size = std::max(
          cur_train_target_bs,
          std::min(target_bs_predict_by_avail_mem, target_bs_calcu_by_delta_mem)
      );
    } else {
      // w/ native gpu memory management
      target_bs_predict_by_avail_mem = -1;

      target_bs_calcu_by_delta_mem =
          cur_train_target_bs + 
          adjuster_->GetDeltaBatchSize(0, std::max(free_mem_mb, 0.0));
      
      target_batch_size = std::max(
          cur_train_target_bs,
          target_bs_calcu_by_delta_mem
      );
    }
  }

  std::vector<TrainAdjuster::AdjustPlan> adjust_plan(train_world_size);

  int min_target_bs = *std::min_element(
      target_batch_sizes.begin(), target_batch_sizes.end());

  if (min_target_bs <= 0 /* avoid partial training workers executing */
      || !adjuster_->CheckImbalance(min_target_bs)) {
    adjuster_->FillSameAdjustPlan(
        AdjustPlan{.batch_size = min_target_bs}, adjust_plan);
  } else {
    adjuster_->FillAdjustPlan(target_batch_sizes, adjust_plan);
  }

  adjuster_->UpdateCachedTrainInfoByAdjustPlan(
      adjust_plan, adjuster_lock);
  adjuster_lock.unlock();

  if (Config::log_memory_adjust) {
    std::stringstream ss;
    ss << "[InferReleaseMemAdjust]"
      << " cur_train_target_bs [" << cur_train_target_batches << "]"
      << " cold cache reserve memory [" << reserve_mem_mbs << "]"
      << " train_avail_mem_mb [" << train_avail_mem_mbs << "]";
    if (Config::use_shared_tensor) {
      ss << " (target_bs_predict_by_avail_mem [" << target_bs_predict_by_avail_mems << "]"
         << " target_bs_calcu_by_delta_mem [" << target_bs_calcu_by_delta_mems << "])";
    } else {
      ss << " (target_bs_calcu_by_delta_mem [" << target_bs_calcu_by_delta_mems << "]"
         << ", not use avail memory to predict bs)";
    }
    ss << " free memory [" << free_mem_mbs << "]"
      << " | release adjust plan: " << PrettyPrintAdjustPlans(adjust_plan);

    LOG(INFO) << ss.str();
  }

  return adjust_plan;
}

int TrainAdjuster::PredictTargetBatchSize(int device_id, memory_mb_t memory_mb) {
  auto [base, slope] = GetModelMemParam();
  auto inftra_info = ctrl::InfTraCommunicator::GetSinfo();
  auto max_batch_size = 
      inftra_info->GetTrainInfoUnsafe(ctrl::kTraRank_0)->init_batch_size;
  
  auto ret = static_cast<int>((memory_mb - base) / slope);
  ret = std::min(ret, max_batch_size);
  ret = std::max(ret, 0);
  // LOG(INFO) << "## " << memory_mb << " " << base << " " << slope
  //           << " " << job_batch_size_;
  return ret;
}

int TrainAdjuster::GetDeltaBatchSize(int device_id, memory_mb_t memory_mb) {
  auto [base, slope] = GetModelMemParam();
  return static_cast<int>(std::ceil(memory_mb / slope));
}

void TrainAdjuster::UpdateCachedTrainInfoByAdjustPlan(
    const std::vector<AdjustPlan> &adjust_plans,
    std::unique_lock<std::mutex> &lock) {
  CHECK(adjust_plans.size() == cached_train_world_size_);
  for (auto rank : boost::irange(adjust_plans.size())) {
    cached_target_batch_sizes_[rank] = 
        adjust_plans[rank].batch_size;
    ctrl::InfTraCommunicator::GetSinfo()->GetMutableTrainInfoUnsafe(rank)
        ->target_batch_size_unpublished = adjust_plans[rank].batch_size;
  }
}

bool TrainAdjuster::CheckImbalance(int proposed_same_batch_size) {
  if (!Config::enable_train_adjust_balance) {
    return false;
  }

  auto train_world_size = cached_train_world_size_;
  std::vector<memory_mb_t> train_avail_mem_mb_minus_cold_cache_reserve;
  for (auto rank : boost::irange(train_world_size)) {
    train_avail_mem_mb_minus_cold_cache_reserve.push_back(
        GetTrainAvailMemMBMinusColdCacheReserve(rank, false));
  }

  memory_mb_t max_waste_mem_mb = 0;
  for (auto rank : boost::irange(train_world_size)) {
    memory_mb_t train_pred_mem_mb = 
      adjuster_->PredictTrainMemUsageMB(rank, proposed_same_batch_size, false);
    auto waste_mem_mb = std::max(0.0, 
        train_avail_mem_mb_minus_cold_cache_reserve[rank] - train_pred_mem_mb);
    max_waste_mem_mb = std::max(max_waste_mem_mb, waste_mem_mb);
  }
  
  LOG_IF(INFO, Config::log_memory_adjust) 
      << "[CheckImbalance] "
      << "proposed_same_batch_size " << proposed_same_batch_size
      << " max_waste_mem_mb " << max_waste_mem_mb
      << " train_avail_mem_mb_minus_cold_cache_reserve "
      << "[" << train_avail_mem_mb_minus_cold_cache_reserve << "]";

  return max_waste_mem_mb > Config::train_adjust_balance_threshold;
}

void TrainAdjuster::FillAdjustPlan(const std::vector<int> &target_batch_sizes, 
                                   std::vector<AdjustPlan> &plans) {
  CHECK(plans.size() == cached_train_world_size_);
  CHECK(target_batch_sizes.size() == cached_train_world_size_);

  for (auto i : boost::irange(cached_train_world_size_)) {
    plans[i] = AdjustPlan{.batch_size = target_batch_sizes[i]};
  }
}

void TrainAdjuster::FillSameAdjustPlan(AdjustPlan plan, 
                                       std::vector<AdjustPlan> &plans) {
  CHECK(plans.size() == cached_train_world_size_);

  for (auto i : boost::irange(cached_train_world_size_)) {
    plans[i] = plan;
  }
}

void TrainAdjuster::FillAdjustPlanOnlyAdjustOne(int rank, AdjustPlan plan, 
                                                std::vector<AdjustPlan> &plans) {
  CHECK(plans.size() == cached_train_world_size_);

  for (auto i : boost::irange(cached_train_world_size_)) {
    plans[i] = (
      i == rank 
      ? plan 
      : AdjustPlan{.batch_size = cached_target_batch_sizes_[i]}
    );
  }
}

memory_mb_t TrainAdjuster::GetTrainAvailMemMBMinusColdCacheReserve(
    int device_id, bool verbose) {
  auto train_avail_mem_mb = 
      std::max(ResourceManager::GetTrainAvailMemoryMB(device_id, verbose), 0.0);
  auto cold_cache_reserve_mem_mb = 
      ColdModelCache::Get(device_id)->GetAdjustReserveMemoryMBUnsafe();

  LOG_IF(INFO, verbose && Config::log_memory_adjust) 
      << "[GetTrainAvailMemMBMinusColdCacheReserve]" 
      << " train avail mem " << train_avail_mem_mb
      << " cold cache reserve mem " << cold_cache_reserve_mem_mb;
  return std::max(train_avail_mem_mb - cold_cache_reserve_mem_mb, 0.0);
}

std::pair<double, double> TrainAdjuster::GetModelMemParam() {
  auto inftra_info = ctrl::InfTraCommunicator::GetSinfo();
  if (!inftra_info->IsTrainInfoValid(ctrl::kTraRank_0)) {
    LOG(FATAL) << "Train info is not valid";
  }
  std::string model_name{
      std::string_view{inftra_info->GetTrainInfoUnsafe(0)->model_name}};
  return GetModelMemParam(model_name);
}

std::pair<double, double> TrainAdjuster::GetModelMemParam(
    const std::string &model_name) {

  // ad-hoc currently, should profile training and return the result
  // to adjuster
  if (Config::use_shared_tensor_train) {
    if (model_name == "resnet152") {
      /*
          8   1.50
         32   3.75
        128  11.75
        150  13.50
      
        AFTER EMPTY CACHE: 0.81 ~ 1.22
      */
      // NOTE: DID NOT consider grad checkpoint
      return {1150, 85};
    } else if (model_name == "swin_b") {
      return {1700, 140};
    } else if (model_name == "swin_b_ddp") {
      // TODO: remove hard-coded value
      return {2560, 140};
      // return {3500, 140};
    } else if (model_name == "gpt2") {
      /*
       
        1   3.00Gb
        4   4.25Gb
        8   6.50Gb
        12  8.75Gb
        16  11.00Gb
        20  12.25Gb

        AFTER EMPTY CACHE: 2.4 ~ 2.9 
       */
      return {2700, 490}; 
    } else if (model_name == "bert_large") {
      LOG(FATAL) << "Unsupported model: " << model_name;
    } else {
      LOG(FATAL) << "Unsupported model: " << model_name;
    }
  } else { // * used for strawman, adjust but no shared tensor
    /*
        8     2.64
        32    4.97
        128  13.75
        150  15.68

        AFTER EMPTY CACHE: 2.28 ~ 2.37
    */
    if (model_name == "resnet152") {
      return {2396, 85};
    } else if (model_name == "swin_b") {
      // NOTE: PLACEHOLDER
      // return {1700, 140};
      return {3300, 145};
    } else {
      LOG(FATAL) << "Unsupported model: " << model_name;
    }
  }
  return {};
}

std::ostream& PrettyPrintAdjustPlans(
    std::ostream &os, 
    const std::vector<TrainAdjuster::AdjustPlan> &plan) {
  os << "target_bs: [ ";
  for (auto &p : plan) {
    os << p.batch_size << " ";
  }
  os << "]";
  return os;
}

std::string PrettyPrintAdjustPlans(
    const std::vector<TrainAdjuster::AdjustPlan> &plan) {
  std::stringstream ss;
  PrettyPrintAdjustPlans(ss, plan);
  return ss.str();
}

} // namespace colserve