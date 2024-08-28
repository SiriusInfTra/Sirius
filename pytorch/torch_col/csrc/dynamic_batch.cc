#include <torch_col/csrc/dynamic_batch.h>
#include <torch_col/csrc/dist_train_sync.h>

#include <boost/range/irange.hpp>

namespace torch_col {

std::unique_ptr<DynamicBatchDistirbutor> 
    DynamicBatchDistirbutor::batch_distributor_ = nullptr;

void DynamicBatchDistirbutor::Init(
    int dataset_size, 
    std::optional<int> global_batch_size) {
  CHECK(batch_distributor_ == nullptr);
  CHECK(DistTrainSync::IsInitialized());
  batch_distributor_ = 
      std::make_unique<DynamicBatchDistirbutor>(
          dataset_size, global_batch_size);
}

DynamicBatchDistirbutor::DynamicBatchDistirbutor(
    int dataset_size, 
    std::optional<int> global_batch_size)
    : dataset_size_(dataset_size), 
      global_batch_size_(global_batch_size) {

  DistTrainSync::CreateCustomSharedData(
    "dist_train_global_shared_data",

    std::make_pair(std::string{"train_batch_cursor"}, 
        &global_shared_data_.train_batch_cursor_), 

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

  NextGlobalBatch();
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

  auto target_bs_unpub = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD_VEC(
      target_batch_size_unpublished);

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
    global_shared_data_.train_batch_cursor_->at(i) = {
        sample_offset, num_samples
    };
    sample_offset += num_samples;

    global_shared_data_.num_unproc_samples_per_train_->at(i) = num_samples;
  }

  // update the batch cursor
  auto it = global_shared_data_.unproc_sample_queue_->begin();
  for (int i = 0; i < train_world_size; i++) {
    int cur_train_num_samples = 0;
    int train_num_samples = 
        global_shared_data_.num_unproc_samples_per_train_->at(i);

    int cursor_left = 0, cursor_right = 0;
    if (it != global_shared_data_.unproc_sample_queue_->end()) {
      cursor_left = it->first;
      cursor_right = it->first;
    } else {
      cursor_left = dataset_size_;
      cursor_right = dataset_size_;
    }

    while (cur_train_num_samples < train_num_samples) {
      int x = GetNumSampleOfBatchIndex(*it);
      CHECK(it != global_shared_data_.unproc_sample_queue_->end());
      if (cur_train_num_samples + x <= train_num_samples) {
        cur_train_num_samples += x;
        cursor_right = it->second;
        it++;
      } else {
        auto [slice, rest] = SliceBatchRange(*it, 
            train_num_samples - cur_train_num_samples);
        global_shared_data_.unproc_sample_queue_->erase(it);

        global_shared_data_.unproc_sample_queue_->insert(slice);
        auto ins_res = 
            global_shared_data_.unproc_sample_queue_->insert(rest);
        it = ins_res.first;
        CHECK(ins_res.second);
        break;
      }
    }

    global_shared_data_.train_batch_cursor_->at(i) = {
        cursor_left, cursor_right
    };
  }
  CHECK(it == global_shared_data_.unproc_sample_queue_->end());

  COMMUNICATOR_UPDATE_SHARED_TRAIN_INFO_FIELD_VEC(
      target_batch_size, target_bs_unpub);
}

std::pair<DynamicBatchDistirbutor::batch_range_vec_t, bool>
DynamicBatchDistirbutor::GetBatch(int batch_size) {
  CHECK(batch_distributor_ != nullptr);

  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  auto &cursor = 
      GLOBAL_SHARED_DATA.train_batch_cursor_->at(train_rank);
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
  
  auto it = batch_distributor_
      ->global_shared_data_.unproc_sample_queue_
      ->lower_bound({cursor.first, 0});
  while (cursor.first < cursor.second
         && num_samples < batch_size) {
    CHECK(it != batch_distributor_
        ->global_shared_data_.unproc_sample_queue_
        ->end());

    int cur_num_sample = 
        batch_distributor_->GetNumSampleOfBatchIndex(*it);
    if (cur_num_sample + num_samples <= batch_size) {
      num_samples += cur_num_sample;
      indices.push_back(*it);
      cursor.first = it->second;

      GLOBAL_SHARED_DATA.procing_sample_queue_->insert(*it);
      it = GLOBAL_SHARED_DATA.unproc_sample_queue_->erase(it);
    } else {
      auto [slice, rest] = batch_distributor_
          ->SliceBatchRange(*it, batch_size - num_samples);
      num_samples += batch_distributor_
          ->GetNumSampleOfBatchIndex(slice);
      indices.push_back(slice);
      cursor.first = slice.second;

      GLOBAL_SHARED_DATA.procing_sample_queue_->insert(slice);
      GLOBAL_SHARED_DATA.unproc_sample_queue_->erase(it);
      auto ins_res =
          GLOBAL_SHARED_DATA.unproc_sample_queue_->insert(rest);
      it = ins_res.first;
    }
    CHECK_LE(cursor.first, cursor.second);
  }

  num_unproc_samples -= num_samples;
  num_procing_samples += num_samples;
  g_num_unproc_samples -= num_samples;
  g_num_procing_samples += num_samples;

  bool require_sync = cursor.first == cursor.second;
  return {indices, require_sync};
}

int DynamicBatchDistirbutor::QueryNextBatchSize() {
  CHECK(batch_distributor_ != nullptr);
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  int train_rank = TorchColConfig::GetTrainRank();

  auto &cursor = 
      GLOBAL_SHARED_DATA.train_batch_cursor_->at(train_rank);
  
  int max_batch_size = 
      GLOBAL_SHARED_DATA.num_unproc_samples_per_train_->at(train_rank);

  int target_batch_size = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
      train_rank, target_batch_size);

  return std::min(max_batch_size, target_batch_size);
}

void DynamicBatchDistirbutor::FinishBatch(
    const batch_range_vec_t &batch_range_vec) {
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
    CHECK(it != GLOBAL_SHARED_DATA.procing_sample_queue_->end());

    GLOBAL_SHARED_DATA.proced_sample_queue_->insert(*it);
    GLOBAL_SHARED_DATA.procing_sample_queue_->erase(it);
  }

  num_procing_samples -= num_samples;
  num_proced_samples += num_samples;
  g_num_procing_samples -= num_samples;
  g_num_proced_samples += num_samples;
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
  // TODO: add 
  bip::scoped_lock lock{*GLOBAL_SHARED_DATA.mut_};
  if (TorchColConfig::IsTrainMaster()) {
    *GLOBAL_SHARED_DATA.num_unproc_samples_ = 
        batch_distributor_->dataset_size_;
    *GLOBAL_SHARED_DATA.num_procing_samples_ = 0;
    *GLOBAL_SHARED_DATA.num_procing_samples_ = 0;

    GLOBAL_SHARED_DATA.unproc_sample_queue_->clear();
    GLOBAL_SHARED_DATA.procing_sample_queue_->clear();
    GLOBAL_SHARED_DATA.proced_sample_queue_->clear();
    GLOBAL_SHARED_DATA.proced_sample_queue_->insert(
        {0, batch_distributor_->dataset_size_}); 

    batch_distributor_->DistributeBatchWithoutLock(false);

    int train_world_size = TorchColConfig::GetTrainWorldSize();
    for (auto i : boost::irange(train_world_size)) {
      GLOBAL_SHARED_DATA.num_procing_samples_per_train_->at(i) = 0;
      GLOBAL_SHARED_DATA.num_proced_samples_per_train_->at(i) = 0;
    }
  }

  DistTrainSync::WaitBarrier();
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

}