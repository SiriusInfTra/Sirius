#pragma once

#include <common/util.h>
#include <common/inf_tra_comm/bip_helper.h>

#include <optional>
#include <mutex>
#include <cstdint>

namespace torch_col {

// [Note: global batch size]
// The global batch size is the effective batch size of updating the model.
// There are three cases of global batch size and grad accumulation:
// - Case 1: w/o grad accumulation
//     training provides the batch size, we calculate the global batch size
//       (global batch size = batch_size * train_world_size),
//     every training iteration updates model
// - Case 2: w/ grad accumulation, fixed batch size
//     training provides the global batch size and micro batch size
// - Case 3: w/ grad accumulation, dynamic batch size
//     training provides the global batch size and initial micro batch size


// [Note: dynamic batch]
// DynamicBatchDistirbutor distribute the micro batch within the global batch
// to balance the workload among devices, i.e., maintaininig the number of 
// sample to be processed by each device of a global batch (global batch size
// is fixed).
// The number of sample of micro batch will be maintained by dataset/data loader.

class DynamicBatchDistirbutor {
 public:
  // each element present sample indices,
  // left is inclusive, right is exclusive
  using batch_range_t = std::pair<int, int>;
  using batch_range_vec_t = std::vector<batch_range_t>;

  static void Init(int dataset_size, 
                   int global_batch_size);
 
  static void DistributeBatch(bool check_num_unproced_samples);

  // Get sample indices for a batch, 
  // return the indices and a bool indicating whether the batch is the last one
  static std::pair<batch_range_vec_t, bool> GetBatch(int batch_size);

  // query the next batch size (the number of samples of next `GetBatch` call)
  static int QueryNextBatchSize();
  
  static void FinishBatch(const batch_range_vec_t &batch_range_vec);
  static void AbortBatch(const batch_range_vec_t &batch_range_vec);

  static void NextGlobalBatch();
  static void NextEpoch();

  DynamicBatchDistirbutor(int dataset_size, 
                          int global_batch_size);

 private:
  static std::unique_ptr<DynamicBatchDistirbutor> batch_distributor_;

  void MergeBatchIndexInQueue(colserve::bip_set<batch_range_t> *queue);
  void DistributeBatchWithoutLock(bool check_num_unproced_samples);

  void NextGlobalBatchImpl();
  void NextEpochImpl();

  int GetNumSampleOfBatchIndex(const batch_range_t &batch_range);

  std::pair<batch_range_t, batch_range_t> 
  SliceBatchRange(const batch_range_t &batch_range, 
                  int num_samples);

  int dataset_size_;
  int global_batch_size_;

  // epoch level
  int num_proced_sample_of_epoch_;
  int num_proced_global_batches_;

  // global batch level
  struct GlobalSharedData {
    std::array<batch_range_t, colserve::MAX_DEVICE_NUM> 
      *train_batch_cursor_;

    std::array<int, colserve::MAX_DEVICE_NUM> 
        *num_unproc_samples_per_train_,
        *num_procing_samples_per_train_,
        *num_proced_samples_per_train_;

    colserve::bip_set<batch_range_t> 
        *unproc_sample_queue_, 
        *procing_sample_queue_,
        *proced_sample_queue_;

    int *num_unproc_samples_, 
        *num_procing_samples_,
        *num_proced_samples_;

    colserve::bip_mutex *mut_;
  } global_shared_data_;

  #define GLOBAL_SHARED_DATA \
    batch_distributor_->global_shared_data_
  
};

} // namespace torch_col