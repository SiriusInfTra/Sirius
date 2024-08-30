#include <torch_col/csrc/dynamic_batch.h>
#include <torch_col/csrc/dist_train_sync.h>

#include <boost/range/irange.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

namespace torch_col {

std::unique_ptr<DynamicBatchDistirbutor> 
    DynamicBatchDistirbutor::batch_distributor_ = nullptr;

void DynamicBatchDistirbutor::Init(
    int dataset_size, 
    int input_batch_size,
    int global_batch_size) {

  CHECK(batch_distributor_ == nullptr);
  CHECK(DistTrainSync::IsInitialized());
  batch_distributor_ = 
      std::make_unique<DynamicBatchDistirbutor>(
          dataset_size, input_batch_size, global_batch_size);
}

DynamicBatchDistirbutor::DynamicBatchDistirbutor(
    int dataset_size, 
    int input_batch_size,
    int global_batch_size)
    : dataset_size_(dataset_size), 
      input_batch_size_(input_batch_size),
      global_batch_size_(global_batch_size),
      num_proced_global_batches_(0),
      num_proced_sample_of_epoch_(0),
      next_epoch_idx_(0),
      num_global_batches_per_epoch_(
        (dataset_size + global_batch_size - 1) / global_batch_size)
{
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

    std::make_pair(std::string{"mut_"},
        &global_shared_data_.mut_)
  );

  // NextEpochImpl();
  // NextGlobalBatchImpl();
}

void DynamicBatchDistirbutor::DistributeBatch(
    bool check_num_unproced_samples) {
  bip::scoped_lock lock{
      *batch_distributor_->global_shared_data_.mut_};

  batch_distributor_
      ->DistributeBatchWithoutLock(check_num_unproced_samples);
}

void DynamicBatchDistirbutor::DistributeBatchWithoutLock(
    bool check_num_unproced_samples) {
  int train_world_size = TorchColConfig::GetTrainWorldSize();

  std::vector<int> target_bs_unpub;
  if (TorchColConfig::HasColocatedInferServer()) {
    target_bs_unpub = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD_VEC(
        target_batch_size_unpublished);
  }

  int num_unprocessed_samples = 0;
  if (check_num_unproced_samples) {
    for (auto i : boost::irange(train_world_size)) {
      num_unprocessed_samples += 
          global_shared_data_.num_unproc_samples_per_train_->at(i);
    }
    CHECK_EQ(num_unprocessed_samples, 
            *global_shared_data_.num_unproc_samples_);
  } else {
    num_unprocessed_samples = 
        *global_shared_data_.num_unproc_samples_;
  }

  int num_samples_per_train = 
      num_unprocessed_samples / train_world_size;
  int num_sample_remainder =
      num_unprocessed_samples % train_world_size;

  // calculate the num of samples for each train
  int sample_offset = 0; 
  for (auto i : boost::irange(train_world_size)) {
    int num_samples = num_samples_per_train + 
        (i < num_sample_remainder ? 1 : 0);
    sample_offset += num_samples;

    global_shared_data_.num_unproc_samples_per_train_->at(i) = num_samples;
  }

  // update the batch cursor
  // auto it = global_shared_data_.unproc_sample_queue_->begin();
  // for (int i = 0; i < train_world_size; i++) {
  //   int cur_train_num_samples = 0;
  //   int train_num_samples = 
  //       global_shared_data_.num_unproc_samples_per_train_->at(i);

  //   int cursor_left = 0, cursor_right = 0;
  //   if (it != global_shared_data_.unproc_sample_queue_->end()) {
  //     cursor_left = it->first;
  //     cursor_right = it->first;
  //   } else {
  //     int last_sample_index = 
  //         num_proced_sample_of_epoch_ + global_batch_size_;
  //     cursor_left = last_sample_index;
  //     cursor_right = last_sample_index;
  //   }

  //   while (cur_train_num_samples < train_num_samples) {
  //     int x = GetNumSampleOfBatchIndex(*it);
  //     CHECK(it != global_shared_data_.unproc_sample_queue_->end());
  //     if (cur_train_num_samples + x <= train_num_samples) {
  //       cur_train_num_samples += x;
  //       cursor_right = it->second;
  //       it++;
  //     } else {
  //       auto [slice, rest] = SliceBatchRange(*it, 
  //           train_num_samples - cur_train_num_samples);
  //       global_shared_data_.unproc_sample_queue_->erase(it);

  //       cursor_right = slice.second;
  //       global_shared_data_.unproc_sample_queue_->insert(slice);
  //       auto ins_res = 
  //           global_shared_data_.unproc_sample_queue_->insert(rest);
  //       it = ins_res.first;
  //       CHECK(ins_res.second);
  //       break;
  //     }
  //   }

  //   // global_shared_data_.train_batch_cursor_->at(i) = {
  //   //     cursor_left, cursor_right
  //   // };
  // }
  // CHECK(it == global_shared_data_.unproc_sample_queue_->end());

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
      << ", ";
    for (auto i : boost::irange(train_world_size)) {
      ss << "train " << i << " num_unproc_samples "
         << global_shared_data_.num_unproc_samples_per_train_->at(i)
         << (i == train_world_size - 1 ? "" : " | ");
    }
    LOG(INFO) << ss.str();
  }

  if (TorchColConfig::HasColocatedInferServer()) {
    COMMUNICATOR_UPDATE_SHARED_TRAIN_INFO_FIELD_VEC(
        target_batch_size, target_bs_unpub);
  }
}

std::pair<DynamicBatchDistirbutor::batch_range_vec_t, bool>
DynamicBatchDistirbutor::GetBatch(int batch_size) {
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

  num_unproc_samples -= num_samples;
  num_procing_samples += num_samples;
  g_num_unproc_samples -= num_samples;
  g_num_procing_samples += num_samples;

  CHECK_GE(num_unproc_samples, 0);
  CHECK_GE(g_num_unproc_samples, 0);

  // bool require_sync = cursor.first == cursor.second;
  bool last_micro_batch = num_unproc_samples == 0;

  LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
      << "[Rank " << TorchColConfig::GetTrainRank() 
      << " | GetBatch] get batch " << indices
      << " batch_size " << (boost::format("%d/%d") % num_samples % batch_size)
      << " num_unproc_samples_per_train " << num_unproc_samples
      << " g_num_procing_samples " << g_num_procing_samples
      << " end_of_global_batch " << last_micro_batch;
      // << " cursor " << cursor;

  return {indices, last_micro_batch};
}

int DynamicBatchDistirbutor::QueryNextBatchSize(
    int batch_size) {
  CHECK(batch_distributor_ != nullptr);
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  // auto &cursor = 
  //     GLOBAL_SHARED_DATA.train_batch_cursor_->at(train_rank);
  
  int max_batch_size = 
      GLOBAL_SHARED_DATA.num_unproc_samples_per_train_->at(train_rank);

  // int target_batch_size = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
  //     train_rank, target_batch_size);

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

  num_procing_samples -= num_samples;
  num_proced_samples += num_samples;
  g_num_procing_samples -= num_samples;
  g_num_proced_samples += num_samples;


  if (end_of_global_batch) {
    CHECK_EQ(num_procing_samples, 0);
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
    const batch_range_vec_t &batch_range_vec) {
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

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
    CHECK(it != GLOBAL_SHARED_DATA.procing_sample_queue_->end());

    GLOBAL_SHARED_DATA.unproc_sample_queue_->insert(*it);
    GLOBAL_SHARED_DATA.procing_sample_queue_->erase(it);
  }

  num_procing_samples -= num_sampels;
  num_unproc_samples += num_sampels;
  g_num_procing_samples -= num_sampels;
  g_num_unproc_sampels += num_sampels;
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
    *global_shared_data_.num_procing_samples_ = 0;

    global_shared_data_.unproc_sample_queue_->clear();
    global_shared_data_.procing_sample_queue_->clear();
    global_shared_data_.proced_sample_queue_->clear();

    if (cur_global_batch_size > 0) {
      global_shared_data_.unproc_sample_queue_->insert(
          {num_proced_sample_of_epoch_, 
          num_proced_sample_of_epoch_ + cur_global_batch_size}); 

      DistributeBatchWithoutLock(false);

      int train_world_size = TorchColConfig::GetTrainWorldSize();
      for (auto i : boost::irange(train_world_size)) {
        // num_unproc be determine in DistributeBatch
        global_shared_data_.num_procing_samples_per_train_->at(i) = 0;
        global_shared_data_.num_proced_samples_per_train_->at(i) = 0;
      }
    } else {
      LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
                << " | NextGlobalBatchImpl] no more samples to process";
      for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
        global_shared_data_.num_unproc_samples_per_train_->at(i) = 0;
        global_shared_data_.num_procing_samples_per_train_->at(i) = 0;
        global_shared_data_.num_proced_samples_per_train_->at(i) = 0;
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

void DynamicBatchDistirbutor::NextEpoch() {
  CHECK(batch_distributor_ != nullptr);

  if (batch_distributor_->first_epoch_start) {
    CHECK_EQ(batch_distributor_->num_proced_sample_of_epoch_, 
        batch_distributor_->dataset_size_);
    CHECK_EQ(batch_distributor_->num_proced_global_batches_, 
        batch_distributor_->num_global_batches_per_epoch_);
  } else {
    batch_distributor_->first_epoch_start = true;
  }
  
  batch_distributor_->NextEpochImpl();
}

void DynamicBatchDistirbutor::NextEpochImpl() {
  // TODO: add 
  LOG_IF(INFO, TorchColConfig::log_dynamic_batch) 
      << "[Rank " << TorchColConfig::GetTrainRank() 
      << " | NextEpochImpl] start new epoch " 
      << next_epoch_idx_;

  next_epoch_idx_ += 1;
  num_proced_global_batches_ = 0;
  num_proced_sample_of_epoch_ = 0;
  // NextGlobalBatchImpl();
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