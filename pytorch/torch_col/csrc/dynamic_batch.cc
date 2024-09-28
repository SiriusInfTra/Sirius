#include <torch_col/csrc/dynamic_batch.h>
#include <torch_col/csrc/dist_train_sync.h>
#include <torch_col/csrc/perf_model.h>

#include <boost/range/irange.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <boost/algorithm/string.hpp>

namespace torch_col {

std::unique_ptr<DynamicBatchDistirbutor> 
    DynamicBatchDistirbutor::batch_distributor_ = nullptr;

DynamicBatchDistirbutor::DistributePolicy 
    DynamicBatchDistirbutor::DISTRIBUTE_POLICY = 
    DynamicBatchDistirbutor::DistributePolicy::SIMPLE;

std::ostream& operator << (
    std::ostream& os, 
    const DynamicBatchDistirbutor::DistributePolicy& policy) {
  switch (policy) {
    case DynamicBatchDistirbutor::DistributePolicy::FIX:
      os << "fix";
      break;
    case DynamicBatchDistirbutor::DistributePolicy::SIMPLE:
      os << "simple";
      break;
    case DynamicBatchDistirbutor::DistributePolicy::BY_PERFORMANCE:
      os << "by_performance";
      break;
    default:
      LOG(FATAL) << "Unknown batch distributing policy " 
                 << static_cast<int>(policy);
      break;
  }
  return os;
}

void DynamicBatchDistirbutor::Init(
    int dataset_size, 
    int input_batch_size,
    int global_batch_size,
    bool lazy_distributing,
    std::string distribute_policy) {

  CHECK(batch_distributor_ == nullptr);
  CHECK(DistTrainSync::IsInitialized());
  
  boost::algorithm::to_lower(distribute_policy);
  if (distribute_policy == "fix") {
    DISTRIBUTE_POLICY = DistributePolicy::FIX;
  } else if (distribute_policy == "simple") {
    DISTRIBUTE_POLICY = DistributePolicy::SIMPLE;
  } else if (distribute_policy == "by_performance") {
    DISTRIBUTE_POLICY = DistributePolicy::BY_PERFORMANCE;
  } else {
    LOG(FATAL) << "Unknown distribute policy " << distribute_policy;
  }

  batch_distributor_ = 
      std::make_unique<DynamicBatchDistirbutor>(
          dataset_size, input_batch_size, 
          global_batch_size, lazy_distributing);
}

DynamicBatchDistirbutor::DynamicBatchDistirbutor(
    int dataset_size, 
    int input_batch_size,
    int global_batch_size,
    bool lazy_distributing)
    : lazy_distributing_(lazy_distributing),
      dataset_size_(dataset_size), 
      input_batch_size_(input_batch_size),
      global_batch_size_(global_batch_size),
      num_proced_global_batches_(0),
      num_proced_sample_of_epoch_(0),
      next_epoch_idx_(0),
      num_global_batches_per_epoch_(
        (dataset_size + global_batch_size - 1) / global_batch_size)
{
  LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
            << " | DynamicBatchDistirbutor] init with dataset_size " 
            << dataset_size << ", input_batch_size " << input_batch_size
            << ", global_batch_size " << global_batch_size
            << ", lazy_distributing " << lazy_distributing
            << ", distributed policy " << DISTRIBUTE_POLICY;

  DistTrainSync::CreateCustomSharedData(
    "dist_train_global_shared_data",

    // std::make_pair(std::string{"train_batch_cursor"}, 
    //     &global_shared_data_.train_batch_cursor_), 

    std::make_pair(std::string{"num_unproc_samples_per_train_"}, 
        &global_shared_data_.num_unproc_samples_per_train_),
    std::make_pair(std::string{"num_procing_samples_per_train_"},
        &global_shared_data_.num_procing_samples_per_train_),
    std::make_pair(std::string{"num_proced_samples_per_train_"},
        &global_shared_data_.num_proced_samples_per_train_),
    
    std::make_pair(std::string{"has_unproc_batches_"},
        &global_shared_data_.has_unproc_batches_),

    std::make_pair(std::string{"unprocessed_samples_"},
        &global_shared_data_.unproc_sample_queue_),
    std::make_pair(std::string{"processing_samples_"},
        &global_shared_data_.procing_sample_queue_),
    std::make_pair(std::string{"processed_samples_"},
        &global_shared_data_.proced_sample_queue_),

    std::make_pair(std::string{"num_unprocessed_samples_"},
        &global_shared_data_.num_unproc_samples_),
    std::make_pair(std::string{"num_processing_samples_"},
        &global_shared_data_.num_procing_samples_),
    std::make_pair(std::string{"num_processed_samples_"},
        &global_shared_data_.num_proced_samples_),

    // std::make_pair(std::string{"last_micro_batch_finish_vote_"},
    //     &global_shared_data_.last_micro_batch_finish_vote_),
    std::make_pair(std::string{"last_micro_batch_finish_vote_cnt_"},
      &global_shared_data_.last_micro_batch_finish_vote_cnt_),
    std::make_pair(std::string{"last_micro_batch_finish_vote_cv_"},
        &global_shared_data_.last_micro_batch_finish_vote_cv_),

    std::make_pair(std::string{"mut_"},
        &global_shared_data_.mut_)
  );

  // NextEpochImpl();
  // NextGlobalBatchImpl();
  if (lazy_distributing_) {
    if (TorchColConfig::GetTrainRank() == 0) {
      for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
        global_shared_data_.num_unproc_samples_per_train_->at(i) = -1;
        global_shared_data_.num_procing_samples_per_train_->at(i) = -1;
        global_shared_data_.num_proced_samples_per_train_->at(i) = -1;
      }
    }
    DistTrainSync::WaitBarrier();
  }
}

void DynamicBatchDistirbutor::DistributeBatch(
    bool check_num_unproced_samples,
    bool distribute_to_all,
    bool at_global_batch_begin) {
  bip::scoped_lock lock{
      *batch_distributor_->global_shared_data_.mut_};

  batch_distributor_
      ->DistributeBatchWithoutLock(check_num_unproced_samples,
                                   distribute_to_all,
                                   at_global_batch_begin);
}

void DynamicBatchDistirbutor::DistributeBatchWithoutLock(
    bool check_num_unproced_samples,
    bool distribute_to_all,
    bool at_global_batch_begin) {
  int train_world_size = TorchColConfig::GetTrainWorldSize();

  std::vector<int> target_bs_unpub;
  if (TorchColConfig::HasColocatedInferServer()) {
    target_bs_unpub = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD_VEC(
        target_batch_size_unpublished);
  }

  bool do_distribute_batch = !(lazy_distributing_ ||
      (!at_global_batch_begin && DISTRIBUTE_POLICY == DistributePolicy::FIX));

  int num_unprocessed_samples = 0;
  if (check_num_unproced_samples && do_distribute_batch) {
    for (auto i : boost::irange(train_world_size)) {
      num_unprocessed_samples += 
          global_shared_data_.num_unproc_samples_per_train_->at(i);
    }
    CHECK_EQ(num_unprocessed_samples, 
            *global_shared_data_.num_unproc_samples_)
      << "num_unproc_samples_per_train_ "
      << boost::accumulate(boost::irange(train_world_size), 
          std::string{}, 
          [&](std::string &acc, int i) {
            auto num = global_shared_data_.
                num_unproc_samples_per_train_->at(i);
            return acc + " " + std::to_string(num);
          });
  } else {
    num_unprocessed_samples = 
        *global_shared_data_.num_unproc_samples_;
  }

  if (distribute_to_all) {
    for (auto i : boost::irange(train_world_size)) {
      global_shared_data_.has_unproc_batches_->at(i) = true;
    }
  }

  if (do_distribute_batch) {
    // we need to skip the worker that has gotten the last micro batch
    std::vector<int> ongoing_workers, 
                     target_bs_vec;
    for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
      if (global_shared_data_.has_unproc_batches_->at(i)) {
        ongoing_workers.push_back(i);
        target_bs_vec.push_back(target_bs_unpub[i]);
      }
    }

    if (ongoing_workers.size() > 0) {
      DistributeBatchImpl(ongoing_workers, target_bs_vec, 
                          num_unprocessed_samples);
    } else {
      CHECK_EQ(num_unprocessed_samples, 0);
    }
  }

  CHECK_GE(num_unprocessed_samples, 0)
      << "g_num_unproc_samples " 
      << *global_shared_data_.num_unproc_samples_ 
      << ", num_unproc_samples_per_train "
      << boost::accumulate(boost::irange(train_world_size), 
          std::string{}, 
          [&](std::string &acc, int i) {
            auto num = global_shared_data_.
                num_unproc_samples_per_train_->at(i);
            return acc + " " + std::to_string(num);
          });

  if (TorchColConfig::log_dynamic_batch) {
    std::stringstream ss;
    ss << "[Rank " << TorchColConfig::GetTrainRank() 
      << " | DistributeBatchWithoutLock]"
      << " num_proced_global_batches "
      << num_proced_global_batches_
      << " g_num_unproc_samples " << num_unprocessed_samples
      << " | train_num_unproc_samples ";
    for (auto i : boost::irange(train_world_size)) {
      ss << " " << global_shared_data_
          .num_unproc_samples_per_train_->at(i);
    }
    LOG(INFO) << ss.str();
  }

  if (TorchColConfig::HasColocatedInferServer()) {
    COMMUNICATOR_UPDATE_SHARED_TRAIN_INFO_FIELD_VEC(
        target_batch_size, target_bs_unpub);
  }
}

void DynamicBatchDistirbutor::DistributeBatchImpl(
    const std::vector<int> &ongoing_workers,
    const std::vector<int> &target_bs_vec,
    int num_unprocessed_samples) {
  CHECK_GT(ongoing_workers.size(), 0);

  if (DISTRIBUTE_POLICY == DistributePolicy::FIX
      || DISTRIBUTE_POLICY == DistributePolicy::SIMPLE) {
    int num_samples_per_train = 
        num_unprocessed_samples / ongoing_workers.size();
    int num_sample_remainder =
        num_unprocessed_samples % ongoing_workers.size();

    int sample_offset = 0;
    for (int i = 0; i < ongoing_workers.size(); ++i) {
      int worker_id = ongoing_workers[i];
      int num_samples = num_samples_per_train 
          + (i < num_sample_remainder ? 1 : 0);

      global_shared_data_
          .num_unproc_samples_per_train_->at(worker_id) = num_samples;
    }
  } else { // BY_PERFORMANCE
    auto perf_vec = PerfModel::GetThptVec(target_bs_vec);
    double perf_sum = std::accumulate(perf_vec.begin(), perf_vec.end(), 0.0);

    LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
        << "[Rank " << TorchColConfig::GetTrainRank() 
        << " | DistributeBatchImpl] target_bs_vec "
        <<  target_bs_vec << " perf_vec " << perf_vec;
    
    int num_distributed_samples = 0;
    for (int i = 0; i < ongoing_workers.size(); ++i) {
      int worker_id = ongoing_workers[i];
      double ratio = perf_vec[i] / perf_sum;
      int num_samples = std::max(1, 
          static_cast<int>(num_unprocessed_samples * ratio));
      global_shared_data_
          .num_unproc_samples_per_train_->at(worker_id) = num_samples;
      num_distributed_samples += num_samples;
    }

    CHECK_LE(num_distributed_samples, num_unprocessed_samples);
    int diff = num_unprocessed_samples - num_distributed_samples;
    for (int i = 0; diff > 0; ++i) {
      if (i >= ongoing_workers.size()) { i = 0; }

      int worker_id = ongoing_workers[i];
      if (global_shared_data_.num_unproc_samples_per_train_->at(worker_id) > 1) {
        global_shared_data_
            .num_unproc_samples_per_train_->at(worker_id) += 1;
        diff--;
      }
    }
  }
}

std::pair<DynamicBatchDistirbutor::batch_range_vec_t, bool>
DynamicBatchDistirbutor::GetBatch(int batch_size_) {
  CHECK(batch_distributor_ != nullptr);

  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  // auto &cursor = 
  //     GLOBAL_SHARED_DATA.train_batch_cursor_->at(train_rank);
  auto &num_unproc_samples = 
      GLOBAL_SHARED_DATA.num_unproc_samples_per_train_->at(train_rank);
  auto &num_procing_samples = 
      GLOBAL_SHARED_DATA.num_procing_samples_per_train_->at(train_rank);

  auto &g_num_unproc_samples = 
      *GLOBAL_SHARED_DATA.num_unproc_samples_;
  auto &g_num_procing_samples = 
      *GLOBAL_SHARED_DATA.num_procing_samples_;

  int batch_size = batch_size_;
  bool last_micro_batch;
  if (batch_distributor_->lazy_distributing_) {
    auto bs = batch_distributor_->LazyDistributingGetBatchSize(
        train_rank, batch_size);
    batch_size = bs.first;
    last_micro_batch = bs.second;
  } else {
    batch_size = std::min(batch_size, num_unproc_samples);  
  }

  int num_samples = 0;
  batch_range_vec_t indices;
  
  // auto it = batch_distributor_
  //     ->global_shared_data_.unproc_sample_queue_
  //     ->lower_bound({cursor.first, 0});
  auto it = batch_distributor_
      ->global_shared_data_.unproc_sample_queue_->begin();
  while (num_samples < batch_size) {
    CHECK(it != batch_distributor_
        ->global_shared_data_.unproc_sample_queue_
        ->end());

    int cur_num_sample = 
        batch_distributor_->GetNumSampleOfBatchIndex(*it);
    if (cur_num_sample + num_samples <= batch_size) {
      num_samples += cur_num_sample;
      indices.push_back(*it);
      // cursor.first = it->second;

      GLOBAL_SHARED_DATA.procing_sample_queue_->insert(*it);
      it = GLOBAL_SHARED_DATA.unproc_sample_queue_->erase(it);
    } else {
      auto [slice, rest] = batch_distributor_
          ->SliceBatchRange(*it, batch_size - num_samples);
      num_samples += batch_distributor_
          ->GetNumSampleOfBatchIndex(slice);
      indices.push_back(slice);
      // cursor.first = slice.second;

      GLOBAL_SHARED_DATA.procing_sample_queue_->insert(slice);
      GLOBAL_SHARED_DATA.unproc_sample_queue_->erase(it);
      auto ins_res =
          GLOBAL_SHARED_DATA.unproc_sample_queue_->insert(rest);
      it = ins_res.first;
    }
    // CHECK_LE(cursor.first, cursor.second);
  }

  if (!batch_distributor_->lazy_distributing_) {
    num_unproc_samples -= num_samples;
    num_procing_samples += num_samples;
    CHECK_GE(num_unproc_samples, 0);
  }
  
  g_num_unproc_samples -= num_samples;
  g_num_procing_samples += num_samples;
  CHECK_GE(g_num_unproc_samples, 0);

  // bool require_sync = cursor.first == cursor.second;
  if (!batch_distributor_->lazy_distributing_) {
    last_micro_batch = num_unproc_samples == 0;
  }
  
  if (last_micro_batch) {
    GLOBAL_SHARED_DATA
        .has_unproc_batches_->at(train_rank) = false;
  }

  LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
      << "[Rank " << TorchColConfig::GetTrainRank() 
      << " | GetBatch] global_batch_idx " 
      << batch_distributor_->num_proced_global_batches_
      << " get batch " << indices
      << " batch_size " << (boost::format("%d/%d") % num_samples % batch_size_)
      << " num_unproc_samples_per_train " << num_unproc_samples
      << " | g_num_procing_samples " << g_num_procing_samples
      << " g_num_unproc_samples " << g_num_unproc_samples
      << " | end_of_global_batch " << last_micro_batch;
      // << " cursor " << cursor;

  return {indices, last_micro_batch};
}

std::pair<int, bool> 
DynamicBatchDistirbutor::LazyDistributingGetBatchSize(
    int train_rank, int batch_size) {
  auto &g_num_unproc_samples = 
      *GLOBAL_SHARED_DATA.num_unproc_samples_;
  auto &g_num_procing_samples = 
      *GLOBAL_SHARED_DATA.num_procing_samples_;

  std::vector<int> ongoing_workers;
  for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
    if (GLOBAL_SHARED_DATA.has_unproc_batches_->at(i)) {
      ongoing_workers.push_back(i);
    }
  }
  CHECK_GT(ongoing_workers.size(), 0);

  int max_batch_size = g_num_unproc_samples - 
      ongoing_workers.size() + 1;
  
  if (batch_size >= max_batch_size) {
    return {max_batch_size, true};
  } else {
    return {batch_size, false};
  }
}

int DynamicBatchDistirbutor::QueryNextBatchSize(
    int batch_size) {
  CHECK(batch_distributor_ != nullptr);
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  // auto &cursor = 
  //     GLOBAL_SHARED_DATA.train_batch_cursor_->at(train_rank);
  
  int max_batch_size;
  if (!batch_distributor_->lazy_distributing_) {
    max_batch_size = 
        GLOBAL_SHARED_DATA.num_unproc_samples_per_train_->at(train_rank);
    // int target_batch_size = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
    //     train_rank, target_batch_size);
  } else {
    max_batch_size = *GLOBAL_SHARED_DATA.num_unproc_samples_;  
  } 

  return std::min(max_batch_size, batch_size);
}

void DynamicBatchDistirbutor::FinishBatch(
    const batch_range_vec_t &batch_range_vec,
    bool end_of_global_batch) {
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  auto &num_procing_samples = 
      GLOBAL_SHARED_DATA.num_procing_samples_per_train_->at(train_rank);
  auto &num_proced_samples =
      GLOBAL_SHARED_DATA.num_proced_samples_per_train_->at(train_rank);

  auto &g_num_procing_samples = 
      *GLOBAL_SHARED_DATA.num_procing_samples_;
  auto &g_num_proced_samples =
      *GLOBAL_SHARED_DATA.num_proced_samples_;
  
  int num_samples = 0;
  for (auto batch_range : batch_range_vec) {
    num_samples += batch_distributor_
        ->GetNumSampleOfBatchIndex(batch_range);
    auto it = GLOBAL_SHARED_DATA.procing_sample_queue_
        ->find(batch_range);
    CHECK(it != GLOBAL_SHARED_DATA.procing_sample_queue_->end())
        << batch_distributor_->PrintBatchQueue(
            GLOBAL_SHARED_DATA.procing_sample_queue_);

    GLOBAL_SHARED_DATA.proced_sample_queue_->insert(*it);
    GLOBAL_SHARED_DATA.procing_sample_queue_->erase(it);
  }

  if (!batch_distributor_->lazy_distributing_) {
    num_procing_samples -= num_samples;
    num_proced_samples += num_samples;
  }

  g_num_procing_samples -= num_samples;
  g_num_proced_samples += num_samples;


  if (end_of_global_batch) {
    CHECK(batch_distributor_->lazy_distributing_ 
          || num_procing_samples == 0) << num_procing_samples;
    batch_distributor_->num_proced_global_batches_++;

    int num_proced_sample_of_epoch = (
      batch_distributor_->num_proced_sample_of_epoch_ 
      + batch_distributor_->global_batch_size_
    );
    batch_distributor_->num_proced_sample_of_epoch_ = 
        std::min(num_proced_sample_of_epoch, 
                 batch_distributor_->dataset_size_);
  }

  LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
      << "[Rank " << TorchColConfig::GetTrainRank() 
      << " | FinishBatch] finish batch " << batch_range_vec
      << " end_of_global_batch " << end_of_global_batch
      << " num_proced_samples_per_train "
      << num_proced_samples
      << " num_procing_samples_per_train " 
      << num_procing_samples
      << " | g_num_proced_samples " << g_num_proced_samples
      << " g_num_procing_samples " << g_num_procing_samples;
}

void DynamicBatchDistirbutor::AbortBatch(
    const batch_range_vec_t &batch_range_vec, 
    bool end_of_global_batch) {
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  LOG_IF(INFO, TorchColConfig::log_dynamic_batch)
      << "[Rank " << TorchColConfig::GetTrainRank()
      << " | AbortBatch] abort batch " << batch_range_vec
      << " end_of_global_batch " << end_of_global_batch;

  auto &num_procing_samples = 
      GLOBAL_SHARED_DATA.num_procing_samples_per_train_->at(train_rank);
  auto &num_unproc_samples =
      GLOBAL_SHARED_DATA.num_unproc_samples_per_train_->at(train_rank);

  auto &g_num_procing_samples =
      *GLOBAL_SHARED_DATA.num_procing_samples_;
  auto &g_num_unproc_sampels =
      *GLOBAL_SHARED_DATA.num_unproc_samples_;
  
  int num_sampels = 0;
  for (auto batch_range : batch_range_vec) {
    num_sampels += batch_distributor_
        ->GetNumSampleOfBatchIndex(batch_range);
    auto it = GLOBAL_SHARED_DATA.procing_sample_queue_
        ->find(batch_range);
    CHECK(it != GLOBAL_SHARED_DATA.procing_sample_queue_->end())
        << "batch_range " << batch_range;

    GLOBAL_SHARED_DATA.unproc_sample_queue_->insert(*it);
    GLOBAL_SHARED_DATA.procing_sample_queue_->erase(it);
  }

  if (!batch_distributor_->lazy_distributing_) {
    num_procing_samples -= num_sampels;
    num_unproc_samples += num_sampels;
  }
  
  g_num_procing_samples -= num_sampels;
  g_num_unproc_sampels += num_sampels;
}

bool DynamicBatchDistirbutor::VoteFinishLastMicroBatch() {
  CHECK(batch_distributor_ != nullptr);
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};

  // int vote_cnt = batch_distributor_
  //     ->GetLastMicroBatchFinishVoteWithoutLock();
  // GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_
  //     ->at(TorchColConfig::GetTrainRank()) = VOTE_FINISH_LAST_MICRO_BATCH;
  int vote_cnt = GLOBAL_SHARED_DATA
      .last_micro_batch_finish_vote_cnt_->fetch_add(
        VOTE_FINISH_LAST_MICRO_BATCH, std::memory_order_relaxed);
    
  if (vote_cnt + VOTE_FINISH_LAST_MICRO_BATCH 
      == TorchColConfig::GetTrainWorldSize()) {
    GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_cv_->notify_all();
    return true;
  } else if (vote_cnt < 0) {
    return false;
  }

  vote_cnt = 0;
  GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_cv_->wait(lock, [&]() {
      vote_cnt = batch_distributor_
          ->GetLastMicroBatchFinishVoteWithoutLock();
      return vote_cnt == TorchColConfig::GetTrainWorldSize() 
          || vote_cnt < 0;
    });

  if (vote_cnt < 0) {
    return false;
  } else {
    return true;
  }
}

void DynamicBatchDistirbutor::VoteAbortLastMicroBatch() {
  CHECK(batch_distributor_ != nullptr);
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};

  GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_cnt_->store(
      ABORT_LAST_MICRO_BATCH, std::memory_order_relaxed);
  // GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_
  //     ->at(TorchColConfig::GetTrainRank()) = ABORT_LAST_MICRO_BATCH;
  GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_cv_->notify_all();
}

void DynamicBatchDistirbutor::ResetLastMicroBatchFinishVote() {
  CHECK(batch_distributor_ != nullptr);
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  batch_distributor_->ResetLastMicroBatchFinishVoteWithLock();
}

void DynamicBatchDistirbutor::NextGlobalBatch() {
  CHECK(batch_distributor_ != nullptr);
  batch_distributor_->NextGlobalBatchImpl();
}

void DynamicBatchDistirbutor::NextGlobalBatchImpl() {
  if (TorchColConfig::log_dynamic_batch) {
    bip::scoped_lock lock{*global_shared_data_.mut_};
    LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
              << " | NextGlobalBatchImpl]"
              << " num_proced_global_batches " << num_proced_global_batches_
              << " last global batch stat: num_proced_samples_per_train "
              << global_shared_data_.num_proced_samples_per_train_->at(
                    TorchColConfig::GetTrainRank());
  }

  // TODO: consider imbalance of resources and its impact on
  //   the training thpt 

  // guarantee all train finish the global batch
  DistTrainSync::WaitBarrier();

  if (TorchColConfig::IsTrainMaster()) {
    bip::scoped_lock lock{*global_shared_data_.mut_};

    int train_world_size = TorchColConfig::GetTrainWorldSize();
    // CHECK_EQ(GLOBAL_SHARED_DATA
    //     .last_micro_batch_finish_vote_cnt_->load(std::memory_order_relaxed),
    //     TorchColConfig::GetTrainWorldSize());

    batch_distributor_->ResetLastMicroBatchFinishVoteWithLock();

    int cur_global_batch_size = 
        global_batch_size_ + num_proced_sample_of_epoch_ > dataset_size_ 
        ? dataset_size_ - num_proced_sample_of_epoch_
        : global_batch_size_;

    CHECK_GE(cur_global_batch_size, 0) 
        << "[Rank " << TorchColConfig::GetTrainRank() 
        << " | NextGlobalBatchImpl] cur_global_batch_size " 
        << cur_global_batch_size
        << " num_proced_sample_of_epoch " 
        << num_proced_sample_of_epoch_;

    *global_shared_data_.num_unproc_samples_ = cur_global_batch_size;
    *global_shared_data_.num_procing_samples_ = 0;
    *global_shared_data_.num_proced_samples_ = 0;

    global_shared_data_.unproc_sample_queue_->clear();
    global_shared_data_.procing_sample_queue_->clear();
    global_shared_data_.proced_sample_queue_->clear();

    // for (auto i : boost::irange(train_world_size)) {
    //   CHECK(!global_shared_data_.has_unproc_batches_);
    //   global_shared_data_.has_unproc_batches_->at(i) = true;
    // }

    if (cur_global_batch_size > 0) {
      global_shared_data_.unproc_sample_queue_->insert(
          {num_proced_sample_of_epoch_, 
          num_proced_sample_of_epoch_ + cur_global_batch_size}); 

      // DistributeBatchWithoutLock(
      //     false, false /* has_unproc_batches has been set previously */);
      DistributeBatchWithoutLock(false, true, true);

      if (!lazy_distributing_) {
        for (auto i : boost::irange(train_world_size)) {
          // num_unproc be determine in DistributeBatch
          global_shared_data_.num_procing_samples_per_train_->at(i) = 0;
          global_shared_data_.num_proced_samples_per_train_->at(i) = 0;
        }
      }
    } else {
      LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
          << "[Rank " << TorchColConfig::GetTrainRank() 
          << " | NextGlobalBatchImpl] no more samples to process";
      if (!lazy_distributing_) {
        for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
          global_shared_data_.num_unproc_samples_per_train_->at(i) = 0;
          global_shared_data_.num_procing_samples_per_train_->at(i) = 0;
          global_shared_data_.num_proced_samples_per_train_->at(i) = 0;
        }
      }
    }
  }

  DistTrainSync::WaitBarrier();

  if (TorchColConfig::log_dynamic_batch) {
    bip::scoped_lock lock{*global_shared_data_.mut_};
    LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
              << " | NextGlobalBatchImpl]"
              << " next global batch stat: num_un_proc_samples_per_train "
              << global_shared_data_.num_unproc_samples_per_train_->at(
                    TorchColConfig::GetTrainRank());
  }
}

int DynamicBatchDistirbutor::NextEpoch() {
  CHECK(batch_distributor_ != nullptr);

  if (batch_distributor_->first_epoch_start) {
    CHECK_EQ(batch_distributor_->num_proced_sample_of_epoch_, 
        batch_distributor_->dataset_size_);
    CHECK_EQ(batch_distributor_->num_proced_global_batches_, 
        batch_distributor_->num_global_batches_per_epoch_);
  } else {
    batch_distributor_->first_epoch_start = true;
  }
  
  return batch_distributor_->NextEpochImpl();
}

int DynamicBatchDistirbutor::NextEpochImpl() {
  // TODO: add 
  LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
      << "[Rank " << TorchColConfig::GetTrainRank() 
      << " | NextEpochImpl] start new epoch " 
      << next_epoch_idx_;

  auto ret = next_epoch_idx_;

  next_epoch_idx_ += 1;
  num_proced_global_batches_ = 0;
  num_proced_sample_of_epoch_ = 0;
  // NextGlobalBatchImpl();

  return ret;
}

int DynamicBatchDistirbutor::GetNumSampleOfBatchIndex(
    const batch_range_t &batch_range) {
  CHECK_GT(batch_range.second, batch_range.first);
  return batch_range.second - batch_range.first;
}

std::pair<DynamicBatchDistirbutor::batch_range_t,
          DynamicBatchDistirbutor::batch_range_t> 
DynamicBatchDistirbutor::SliceBatchRange(
    const batch_range_t &batch_range, int num_samples) {

  CHECK_GE(num_samples, 0);
  CHECK_LE(num_samples, GetNumSampleOfBatchIndex(batch_range));

  return {
    std::make_pair(batch_range.first, batch_range.first + num_samples),
    std::make_pair(batch_range.first + num_samples, batch_range.second)
  };
}

int DynamicBatchDistirbutor::GetLastMicroBatchFinishVoteWithoutLock() {
  // int vote_cnt = 0;
  // for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
  //   int vote = global_shared_data_
  //     .last_micro_batch_finish_vote_->at(i);
  //   if (vote == ABORT_LAST_MICRO_BATCH) {
  //     return ABORT_LAST_MICRO_BATCH;
  //   }
  //   vote_cnt += vote;
  // }
  // return vote_cnt;
  return GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_cnt_->load(
      std::memory_order_relaxed);
}

void DynamicBatchDistirbutor::ResetLastMicroBatchFinishVoteWithLock() {
  // for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
  //   GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_->at(i) = 0;
  // }
  GLOBAL_SHARED_DATA.last_micro_batch_finish_vote_cnt_->store(
      0, std::memory_order_relaxed);
}

std::string DynamicBatchDistirbutor::PrintBatchQueue(
    const colserve::bip_set<batch_range_t> *queue) {
  std::stringstream ss;
  ss << "{";
  for (auto it = queue->begin(); it != queue->end(); it++) {
    ss << *it << (std::next(it) == queue->end() ? "" : ", ");
  }
  ss << "}";
  return ss.str();
}

} // namespace torch_col