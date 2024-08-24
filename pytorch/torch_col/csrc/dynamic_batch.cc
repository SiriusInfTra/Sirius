#include <torch_col/csrc/dynamic_batch.h>
#include <torch_col/csrc/dist_train_sync.h>

namespace torch_col {

DynamicBatchDistirbutor::DynamicBatchDistirbutor(
    int dataset_size, int global_batch_size)
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

    std::make_pair(std::string{"unprocessed_samples_"},
      &global_shared_data_.unprocessed_samples_),

    std::make_pair(std::string{"processing_samples_"},
      &global_shared_data_.processing_samples_),

    std::make_pair(std::string{"num_unprocessed_samples_"},
      &global_shared_data_.num_unprocessed_samples_),

    std::make_pair(std::string{"num_processing_samples_"},
      &global_shared_data_.num_processing_samples_),

    std::make_pair(std::string{"mut_"},
      &global_shared_data_.mut_)
  );

  if (TorchColConfig::IsTrainMaster()) {

  }
}


void DynamicBatchDistirbutor::DistributeBatch() {

}

std::pair<DynamicBatchDistirbutor::batch_indices_t, bool>
DynamicBatchDistirbutor::GetBatch(int batch_size) {
  
}

void DynamicBatchDistirbutor::CommitBatch(
    const batch_indices_t &samples) {

}

void DynamicBatchDistirbutor::AbortBatch(
    const batch_indices_t &samples) {

}

void DynamicBatchDistirbutor::NextGlobalBatch() {
  // std::unique_lock lock{mut_};
  // unprocessed_samples_.insert({0, dataset_size_});
  // processing_samples_.clear();
}

}