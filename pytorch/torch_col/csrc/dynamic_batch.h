#pragma once

#include <common/util.h>
#include <common/inf_tra_comm/bip_helper.h>

#include <mutex>
#include <cstdint>

namespace torch_col {

enum class BatchDistributorEvent {
  kCommitBatch,
  kAbortBatch,
  k
};


class DynamicBatchDistirbutor {
 public:
  using batch_indices_t = std::vector<std::pair<int, int>>;

  DynamicBatchDistirbutor(int dataset_size, 
                          int global_batch_size);
 
  void DistributeBatch();

  // Get sample indices for a batch, 
  // return the indices and a bool indicating whether the batch is the last one
  std::pair<batch_indices_t, bool> GetBatch(int batch_size);
  
  void CommitBatch(const batch_indices_t &samples);
  void AbortBatch(const batch_indices_t &samples);
  void NextGlobalBatch();

 private:
  int dataset_size_, global_batch_size_;

  struct GlobalSharedData {
    std::array<std::pair<int, int>, colserve::MAX_DEVICE_NUM> 
      *train_batch_cursor_;

    std::array<int, colserve::MAX_DEVICE_NUM> 
        *num_unproc_samples_per_train_,
        *num_procing_samples_per_train_;

    colserve::bip_set<std::pair<int, int>> 
        *unprocessed_samples_, 
        *processing_samples_;

    int *num_unprocessed_samples_, 
        *num_processing_samples_;

    colserve::bip_mutex *mut_;
  } global_shared_data_;
  
};

} // namespace torch_col