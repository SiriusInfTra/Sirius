#include <common/log_as_glog_sta.h>
#include <common/sm_partition.h>
#include <common/util.h>

namespace colserve {

std::unique_ptr<SMPartitioner> SMPartitioner::sm_partitioner_ = nullptr;

void SMPartitioner::Init(int device, bool cleanup) {
  if (sm_partitioner_ == nullptr) {
    sm_partitioner_ = std::make_unique<SMPartitioner>(device, cleanup);
  }
}

SMPartitioner::SMPartitioner(int device, bool cleanup) : device_(device) {
  auto *gpu_id = std::getenv("CUDA_VISIBLE_DEVICES");
  CHECK(gpu_id != nullptr);
  shm_name_ = "gpu-colocate-sm-partition-" + std::to_string(getuid()) + "-" + gpu_id;

  LOG(INFO) << "[SM Partitioner] initialize, shm_name " << shm_name_.c_str();

  if (cleanup) {
    bip::shared_memory_object::remove(shm_name_.c_str());
  }

  shm_ = bip::managed_shared_memory{bip::open_or_create, shm_name_.c_str(), 65536};

  auto atomic_init = [&]() {
    tpc_data_ = shm_.find_or_construct<TpcData>("tpc_data")();
    ref_cnt_ = shm_.find_or_construct<std::atomic<int>>("ref_cnt")();
  };
  shm_.atomic_func(atomic_init);

  ref_cnt_->fetch_add(1, std::memory_order_relaxed);

  if (cleanup) {
    memset(tpc_data_, 0, sizeof(TpcData));
  }

  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  deviceProp.multiProcessorCount;

  gpu_sm_num_ = deviceProp.multiProcessorCount;
  gpu_tpc_num_ = gpu_sm_num_ >> 1;
  sm_num_per_tpc_ = 2; // volta -- hopper
}

SMPartitioner::~SMPartitioner() {
  auto cnt = ref_cnt_->fetch_sub(1, std::memory_order_relaxed);
  if (cnt == 1) {
    bip::shared_memory_object::remove(shm_name_.c_str());
  }
}

int SMPartitioner::GetGPUNumSM() {
  CHECK(sm_partitioner_ != nullptr);
  return sm_partitioner_->gpu_sm_num_;
}

int SMPartitioner::GetGPUNumTpc() {
  CHECK(sm_partitioner_ != nullptr);
  return sm_partitioner_->gpu_tpc_num_;
}

void SMPartitioner::SetInferRequiredTpcNum(int tpc_num) {
  CHECK(sm_partitioner_ != nullptr);
  sm_partitioner_->tpc_data_->infer_required_tpc_num.store(tpc_num, std::memory_order_relaxed);
}

void SMPartitioner::AddInferRequiredTpcNum(int tpc_num) {
  CHECK(sm_partitioner_ != nullptr);
  sm_partitioner_->tpc_data_->infer_required_tpc_num.fetch_add(tpc_num, std::memory_order_relaxed);
}

void SMPartitioner::DecInferRequiredTpcNum(int tpc_num) {
  CHECK(sm_partitioner_ != nullptr);
  sm_partitioner_->tpc_data_->infer_required_tpc_num.fetch_sub(tpc_num, std::memory_order_relaxed);
}

uint64_t SMPartitioner::GetTrainAvailTpcMask() {
  CHECK(sm_partitioner_ != nullptr);
  int num_disabled = sm_partitioner_->tpc_data_->infer_required_tpc_num.load(std::memory_order_relaxed);
  CHECK_GT(num_disabled, 0);
  if (num_disabled == 0) {
    return 0;
  }

  if (num_disabled >= sm_partitioner_->gpu_tpc_num_) {
    // avoid hanging training kernel
    num_disabled = sm_partitioner_->gpu_tpc_num_ - 1; 
  }
  uint64_t mask = (1ull << num_disabled) - 1;
  return mask;
}

int SMPartitioner::GetTrainAvailTpcNum() {
  CHECK(sm_partitioner_ != nullptr);
  int num_disabled = sm_partitioner_->tpc_data_->infer_required_tpc_num.load(std::memory_order_relaxed);
  CHECK_GT(num_disabled, 0);
  return std::max(1, sm_partitioner_->gpu_tpc_num_ - num_disabled);
}

uint64_t SMPartitioner::SetTrainStreamTpcMask(CUstream s) {
  CHECK(sm_partitioner_ != nullptr);
  auto mask = GetTrainAvailTpcMask();
  SetStreamTpcMask(s, mask);
  return mask;
}

}