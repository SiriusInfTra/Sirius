#include <common/log_as_glog_sta.h>
#include <common/sm_partition.h>
#include <common/util.h>
#include <common/inf_tra_comm/communicator.h>

namespace colserve {

std::array<std::unique_ptr<SMPartitioner>, MAX_DEVICE_NUM> 
    SMPartitioner::sm_partitioners_{nullptr};
thread_local std::unordered_map<CUstream, uint64_t> 
    SMPartitioner::stream_last_tpc_mask_map_;

void SMPartitioner::Init(int device_id) {
  // DLOG(INFO) << "[SM Partitioner] Init";
  if (sm_partitioners_[device_id] == nullptr) {
    sm_partitioners_[device_id] = std::make_unique<SMPartitioner>(device_id);
  }
}

SMPartitioner* SMPartitioner::Get(int device_id) {
  CHECK(sm_partitioners_[device_id] != nullptr) << "SMPartitioner not initialized";
  return sm_partitioners_[device_id].get();
}

SMPartitioner::SMPartitioner(int device_id) : device_id_(device_id) {
  LOG(INFO) << "[SM Partitioner] initialize, "
            << "train_tpc " << min_train_tpc_num_ << " -- " << max_train_tpc_num_;
            // << ", "
            // << "shm_name " << shm_name_.c_str();

  tpc_data_ = &ctrl::InfTraCommunicator::GetSinfo()
      ->GetMutableInferInfoUnsafe()->tpc_datas[device_id_];

  cudaDeviceProp deviceProp;
  COL_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_id));
  // deviceProp.multiProcessorCount;

  gpu_sm_num_ = deviceProp.multiProcessorCount;
  gpu_tpc_num_ = gpu_sm_num_ >> 1;
  sm_num_per_tpc_ = 2; // volta -- hopper

  CHECK_GE(min_train_tpc_num_, 1);
  CHECK_LE(max_train_tpc_num_, gpu_tpc_num_);
  CHECK_GE(max_train_tpc_num_, min_train_tpc_num_);
}

SMPartitioner::~SMPartitioner() {
}

int SMPartitioner::GetGPUNumSM() {
  return gpu_sm_num_;
}

int SMPartitioner::GetGPUNumTpc() {
  return gpu_tpc_num_;
}

void SMPartitioner::SetInferRequiredTpcNum(int tpc_num) {
  tpc_data_->infer_required_tpc_num.store(tpc_num, std::memory_order_relaxed);
}

int SMPartitioner::GetInferRequiredTpcNum() {
  return tpc_data_->infer_required_tpc_num.load(std::memory_order_relaxed);
}

void SMPartitioner::AddInferRequiredTpcNum(int tpc_num) {
  tpc_data_->infer_required_tpc_num.fetch_add(tpc_num, std::memory_order_relaxed);
}

void SMPartitioner::DecInferRequiredTpcNum(int tpc_num) {
  tpc_data_->infer_required_tpc_num.fetch_sub(tpc_num, std::memory_order_relaxed);
}

uint64_t SMPartitioner::GetTrainAvailTpcMask() {
  int num_disabled = tpc_data_->infer_required_tpc_num.load(std::memory_order_relaxed);
  CHECK_GE(num_disabled, 0);
  if (num_disabled == 0) {
    return 0;
  }

  // avoid hanging training kernel
  num_disabled = std::min(num_disabled, 
                          gpu_tpc_num_ - min_train_tpc_num_);

  num_disabled = std::max(num_disabled, 
                          gpu_tpc_num_ - max_train_tpc_num_);

  // train use SM with higher index
  uint64_t mask = (1ull << num_disabled) - 1;

  return mask;
}

int SMPartitioner::GetTrainAvailTpcNum() {
  int num_disabled = tpc_data_->infer_required_tpc_num.load(std::memory_order_relaxed);
  CHECK_GE(num_disabled, 0);
  
  int res = std::max(min_train_tpc_num_, 
                     gpu_tpc_num_ - num_disabled);
  res = std::min(res, max_train_tpc_num_);
  return res;
}

uint64_t SMPartitioner::SetTrainStreamTpcMask(CUstream s) {
  auto mask = GetTrainAvailTpcMask();
  SetStreamTpcMask(s, mask);
  return mask;
}

}