#pragma once

#include <common/util.h>
#include <common/inf_tra_comm/bip_helper.h>

#include <mutex>
#include <cstdint>

namespace torch_col {

class DynamicBatchDistirbutor {
 public:
  // each element present sample indices,
  // left is inclusive, right is exclusive
  using batch_index_t = std::pair<int, int>;
  using batch_index_vec_t = std::vector<batch_index_t>;

  static void Init(int dataset_size, 
                   int global_batch_size);
 
  static void DistributeBatch(bool check_num_unproced_samples);

  // Get sample indices for a batch, 
  // return the indices and a bool indicating whether the batch is the last one
  static std::pair<batch_index_vec_t, bool> GetBatch(int batch_size);
  
  static void FinishBatch(const batch_index_vec_t &batch_indices);
  static void AbortBatch(const batch_index_vec_t &batch_indices);
  static void NextGlobalBatch();

  DynamicBatchDistirbutor(int dataset_size, 
                          int global_batch_size);

 private:
  static std::unique_ptr<DynamicBatchDistirbutor> batch_distributor_;


  void MergeBatchIndexInQueue(colserve::bip_set<batch_index_t> *queue);
  void DistributeBatchWithoutLock(bool check_num_unproced_samples);

  int GetNumSampleOfBatchIndex(const batch_index_t &batch_index);

  std::pair<batch_index_t, batch_index_t> 
  SliceBatchIndex(const batch_index_t &batch_index, 
                  int num_samples);

  int dataset_size_, global_batch_size_;

  struct GlobalSharedData {
    std::array<batch_index_t, colserve::MAX_DEVICE_NUM> 
      *train_batch_cursor_;

    std::array<int, colserve::MAX_DEVICE_NUM> 
        *num_unproc_samples_per_train_,
        *num_procing_samples_per_train_,
        *num_proced_samples_per_train_;

    colserve::bip_set<batch_index_t> 
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