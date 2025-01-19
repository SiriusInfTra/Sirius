#include <server/llm/kv_cache_pool.h>
#include <server/config.h>
#include <server/train_adjuster.h>
#include <server/control/controller.h>
#include <server/train_launcher.h>

#include <boost/range/irange.hpp>

#include <c10/core/TensorImpl.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

namespace colserve {

std::ostream & operator << (
    std::ostream &os,
    const KVCachePool::KVCacheBlockGroup&kvc_page_group) {
  os << "stat " << kvc_page_group.stat.allocted
      << " " << kvc_page_group.stat.num_idle_blks
      << " " << kvc_page_group.stat.num_used_blks
      << ", blk_idx " << kvc_page_group.block_indices[0]
      << "..." << kvc_page_group.block_indices.back()
      << ", base_addr (k,v) " << std::hex 
      << "0x" << kvc_page_group.layer_kvc_page_base_addr[0].first
      << " 0x" << kvc_page_group.layer_kvc_page_base_addr[0].second
      << std::dec;
  return os;
}

std::unique_ptr<KVCachePool> KVCachePool::kv_cache_pool_ = nullptr;
uint64_t KVCachePool::cls_kv_cache_block_nbytes_ = 0;

void KVCachePool::Init() {
  CHECK(sta::CUDAMemPool::IsEnable());
  kv_cache_pool_ = std::make_unique<KVCachePool>();
}

void KVCachePool::MaybeSetKVCacheBlockNbytes(
    int64_t block_size,
    int64_t num_layers, 
    int64_t num_heads, 
    int64_t head_size, 
    memory_byte_t block_nbytes) {
  if (KVCachePool::cls_kv_cache_block_nbytes_ == 0) {
    KVCachePool::cls_kv_cache_block_nbytes_ = block_nbytes;
  } else {
    CHECK(KVCachePool::cls_kv_cache_block_nbytes_ == block_nbytes)
        << "block_nbytes " << sta::PrintByte(block_nbytes) 
        << " is not consistent with previous value "
        << sta::PrintByte(KVCachePool::cls_kv_cache_block_nbytes_);
  }

  if (!Config::use_shared_tensor) {
    return;
  }

  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};

  uint64_t layer_k_or_v_blk_nbytes = block_size * num_heads * head_size; // size of k (or v) blk size
  CHECK(PhyPageFactor() * sta::CUDAMemPool::PageNbytes() % layer_k_or_v_blk_nbytes == 0) 
      << "k/v block_nbytes of a layer " << layer_k_or_v_blk_nbytes
      << " is not a factor of " 
      << sta::PrintByte(sta::CUDAMemPool::PageNbytes())
      << " x " << PhyPageFactor();
  if (kv_cache_pool_->cache_block_nbytes_ == 0) {
    kv_cache_pool_->cache_block_nbytes_ = block_nbytes;
    kv_cache_pool_->block_size_ = block_size;
    kv_cache_pool_->num_layer_ = num_layers;
    kv_cache_pool_->num_heads_ = num_heads;
    kv_cache_pool_->head_size_ = head_size;
    LOG(INFO) << "KVCachePool block_nbytes " << sta::PrintByte(block_nbytes)
              << " block_size " << block_size << " num_layers " << num_layers 
              << " num_heads " << num_heads << " head_size " << head_size;
  } else {
    CHECK(kv_cache_pool_->cache_block_nbytes_ == block_nbytes)
        << "block_nbytes " << sta::PrintByte(block_nbytes) 
        << " is not consistent with previous value "
        << sta::PrintByte(kv_cache_pool_->cache_block_nbytes_);
  }
}

int KVCachePool::GetNumGpuKVCacheBlocks() {
  // CHECK(KVCachePool::cls_kv_cache_block_nbytes_ != 0);
  if (!Config::use_shared_tensor) {
    return Config::cuda_memory_pool_gb * 1_GB / KVCachePool::cls_kv_cache_block_nbytes_;
  }

  CHECK(kv_cache_pool_ != nullptr);
  size_t item_size = 2;
  size_t layer_k_or_v_blk_nbytes = (
      item_size * kv_cache_pool_->block_size_ 
      * kv_cache_pool_->num_heads_ 
      * kv_cache_pool_->head_size_);
  CHECK(layer_k_or_v_blk_nbytes != 0);
  size_t blk_grp_size = KVCachePool::KVCPageNybtes() / layer_k_or_v_blk_nbytes;

  size_t kvc_blk_grp_nbytes = 
      layer_k_or_v_blk_nbytes 
      * 2 /* k-cache in a page, v-cache in other page */
      * blk_grp_size 
      * kv_cache_pool_->num_layer_;

  LOG(INFO) << "[GetNumGpuKVCacheBlocks] "
            << "layer_k_or_v_blk_nbytes " << sta::PrintByte(layer_k_or_v_blk_nbytes)
            << " blk_grp_size " << blk_grp_size
            << " kvc_blk_grp_nbytes " << sta::PrintByte(kvc_blk_grp_nbytes);

  // auto ret = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
  //     ->PoolNbytes() / kv_cache_pool_->cache_block_nbytes_;
  auto ret = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
      ->PoolNbytes() / kvc_blk_grp_nbytes * blk_grp_size;
  return ret;
}

PyObject* KVCachePool::InitKVCache(
    int n_layers, const bp::list &shape_, std::string dtype, int64_t itemsize) {
  std::vector<int64_t> shape;
  for (auto i : boost::irange(bp::len(shape_))) {
    shape.push_back(bp::extract<int64_t>(shape_[i]));
  }

  CHECK(kv_cache_pool_ != nullptr);
  //z at::ten
  // torch::from_blob()
  memory_byte_t total_nbytes_per_layer = 1;
  for (auto dim : shape) {
    // block_nbytes_per_layer *= dim;
    total_nbytes_per_layer *= dim;
  }
  CHECK(shape.size() == 3 && shape[0] == 2) << "shape " << shape;
  memory_byte_t block_nbytes_per_layer = shape[2] * shape[0];

  CHECK_EQ(block_nbytes_per_layer, 
    2 
    * kv_cache_pool_->num_heads_ 
    * kv_cache_pool_->block_size_ 
    * kv_cache_pool_->head_size_
  );
  block_nbytes_per_layer *= itemsize;
  total_nbytes_per_layer *= itemsize;
  
  // CHECK(total_nbytes_per_layer % sta::CUDAMemPool::PageNbytes() == 0);
  CHECK(total_nbytes_per_layer % KVCPageNybtes() == 0);
  auto num_blocks_per_page = 
      KVCPageNybtes() / block_nbytes_per_layer 
      * 2 /* k-cache in a page, v-cache in other page */;
  auto num_pages_per_layer = 
      total_nbytes_per_layer / KVCPageNybtes();

  CHECK(num_pages_per_layer % 2 == 0);
  auto num_blk_grps_per_layer = (
    num_pages_per_layer / 2 /* k-cache in a page, v-cache in other page */
  );

  if (kv_cache_pool_->kvc_blk_grp_size_ == 0) {
    kv_cache_pool_->kvc_blk_grp_size_ = num_blocks_per_page;
  } else {
    CHECK_EQ(kv_cache_pool_->kvc_blk_grp_size_, num_blocks_per_page);
  }
  if (kv_cache_pool_->num_kvc_blk_grp_per_layer_ == 0) {
    kv_cache_pool_->num_kvc_blk_grp_per_layer_ = num_blk_grps_per_layer;
  } else {
    CHECK_EQ(kv_cache_pool_->num_kvc_blk_grp_per_layer_, num_blk_grps_per_layer);
  }
  
  DLOG(INFO) << "[InitKVCache] layer " << n_layers
            << " block_nbytes_per_layer " << sta::PrintByte(block_nbytes_per_layer)
            << " total_nbytes_per_layer " << sta::PrintByte(total_nbytes_per_layer)
            << " kvc_blk_grp_size " << kv_cache_pool_->kvc_blk_grp_size_
            << " num_pages_per_layer " << num_pages_per_layer
            << " num_kvc_blk_grp_per_layer " 
            << kv_cache_pool_->num_kvc_blk_grp_per_layer_;

  std::unique_lock lock{kv_cache_pool_->mut_};
  try {
    auto tensor_opt = at::TensorOptions()
        .device(at::kCUDA, sta::DeviceManager::GetCurrentDevice());
    if (dtype == "torch.float32") {
      tensor_opt = tensor_opt.dtype(at::kFloat);
    } else if (dtype == "torch.float16") {
      tensor_opt = tensor_opt.dtype(at::kHalf);
    } else if (dtype == "torch.bfloat16") {
      tensor_opt = tensor_opt.dtype(at::kBFloat16);
    } else {
      LOG(FATAL) << "unsupported dtype " << dtype;
    }
    std::vector<int64_t> tensor_shape = {
      shape[0], shape[1], n_layers, shape[2]
    };
    auto tensor = at::from_blob(sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())->GetBasePtr(sta::MemType::kInfer), tensor_shape, tensor_opt);
    return torch::autograd::utils::wrap(tensor);
  } catch (const std::exception& e) {
    LOG(FATAL) << "InitKVCache failed: " << e.what();
  }
}

size_t KVCachePool::GetNumFreeBlocks() {
  CHECK(kv_cache_pool_ != nullptr);
  return Config::cuda_memory_pool_gb * 1_GB - 9_GB - Config::train_memory_over_predict_mb * 1_MB;
}

void KVCachePool::FreeKVCacheBlock(const bp::list &blk_indices) {
  CHECK(kv_cache_pool_ != nullptr);
  size_t page_nbytes = 5 * sta::CUDAMemPool::PageNbytes();
  std::unique_lock lock{kv_cache_pool_->mut_};
  for (auto k : boost::irange(bp::len(blk_indices))) {
    auto blk_idx = bp::extract<int64_t>(blk_indices[k]);
    auto iter = kv_cache_pool_->kv_cache_blocks_.find(blk_idx);
    CHECK(iter != kv_cache_pool_->kv_cache_blocks_.end());
    kv_cache_pool_->kv_cache_blocks_.erase(iter);
  }
  if (!ctrl::Controller::Get()->IsTrainIdle() &&
    sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())->NumFreePages() * sta::CUDAMemPool::PageNbytes() >= 6_GB) {
    kv_cache_pool_->ReclaimMemToTrain(lock);
  }
}

bp::list KVCachePool::AllocKVCacheBlock(size_t n) {
  CHECK(kv_cache_pool_ != nullptr);
  // NOTE: nbytes = 160MB * n
  size_t page_nbytes = 160_MB;
  auto *mem_pool = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice());
  std::unique_lock lock{kv_cache_pool_->mut_};
  int64_t train_memory = std::max(
    static_cast<size_t>(TrainAdjuster::PredictTrainMemUsageMB(sta::DeviceManager::GetCurrentDevice(), true) * 1_MB),
    mem_pool->TrainAllMemUsage()
  );
  int64_t available_memory = 
    static_cast<int64_t>(mem_pool->NumFreePages() * sta::CUDAMemPool::PageNbytes()) 
    + static_cast<int64_t>(mem_pool->TrainAllMemUsage()) - train_memory 
    - Config::train_memory_over_predict_mb * 1_MB;
  if (available_memory < static_cast<int64_t>(n * page_nbytes)) {
    kv_cache_pool_->MaybeAdjustTrain(
      std::max(
        5_GB / sta::CUDAMemPool::PageNbytes(),
         n * page_nbytes / sta::CUDAMemPool::PageNbytes()
      ), n * page_nbytes / sta::CUDAMemPool::PageNbytes(), lock);
  }
  bp::list ret;
  for (auto k : boost::irange(n)) {
    auto block = mem_pool->Alloc(page_nbytes, sta::MemType::kInfer, false);
    int64_t blk = block->block->addr_offset; 
    CHECK(kv_cache_pool_->kv_cache_blocks_.insert(std::make_pair(blk, block)).second);
    ret.append(blk);
  }
  
  return ret;
}


uint64_t KVCachePool::MaybeAdjustTrain(uint64_t num_required_pages,
                                   uint64_t min_num_required_pages,
                                   std::unique_lock<std::mutex> &kvc_pool_lock) {
  auto device_id = sta::DeviceManager::GetCurrentDevice();
  auto mpool = sta::CUDAMemPool::Get(device_id);
  // const auto page_size = 32_MB;
  // const auto page_size = sta::CUDAMemPool::PageNbytes();
  // uint64_t num_free_pages = mpool->NumFreePages() / PhyPageFactor();
  // if (num_free_pages > num_required_pages) {
  //   return num_free_pages;
  // }
  memory_mb_t require_mb = sta::ByteToMB(
    num_required_pages * sta::CUDAMemPool::PageNbytes()
    + Config::train_over_adjust_nbytes
  );
  LOG_IF(INFO, Config::log_memory_adjust)
      << "[KVCachePool, Memory Adjust] " 
      << " num_required_pages " << num_required_pages
      << ", " << require_mb << " MB";

  // std::mutex fake_mut;
  // std::unique_lock fake_lock{fake_mut, std::defer_lock};
  // auto adjust_plan = TrainAdjuster::GetInferRequireMemAdjustPlanWithInLock(
  //     device_id, require_mb, std::nan(""), fake_lock);
  auto adjust_plan = TrainAdjuster::GetLLMInferRequireMemAdjustPlan(
      device_id, require_mb, kvc_pool_lock);
  if (!adjust_plan.empty()) {
    auto t0 = Profiler::Now();
    PROFILE_START(TrainAdjust);
    auto cmd_id = ctrl::Controller::Get()->ColocateInferRequireAdjust(
        0, device_id, adjust_plan);
    ctrl::Controller::Get()->WaitColocateAdjustDone(cmd_id);
    PROFILE_END(TrainAdjust);
    LOG_IF(INFO, Config::log_memory_adjust) 
        << "[KVCachePool, Memory Adjust] "
        << "AllocStorageMaybeAdjust: wait adjust" 
        << " num_free_pages(mpool) " << mpool->NumFreePages()
        << " " << PROFILE_DURATRION(TrainAdjust) << "ms";
  }
  uint64_t num_free_pages_after = mpool->NumFreePages();
  CHECK_GE(num_free_pages_after, min_num_required_pages);
  return num_free_pages_after;
}

void KVCachePool::ReclaimMemToTrain(
    std::unique_lock<std::mutex> &kvc_pool_lock) {
  auto adjust_plan = TrainAdjuster::GetLLMInferReleaseMemAdjustPlan(
      kvc_pool_lock);
  if (adjust_plan.empty()) {
    return;
  }
  ctrl::Controller::Get()->ColocateInferReleaseAdjust(adjust_plan);
}

double KVCachePool::GetKVCacheMemPageUtil() {
  auto & kvc_pool_stat = kv_cache_pool_->kvc_pool_stat_;
  if (kvc_pool_stat.num_allocated_blk_grps_.load(std::memory_order_relaxed) == 0) {
    return std::nan("");
  }
  return 1 - (
    1.0 * kvc_pool_stat.num_idle_blk_grps_.load(std::memory_order_relaxed) / 
        kvc_pool_stat.num_allocated_blk_grps_.load(std::memory_order_relaxed)
  );
}

KVCachePoolStat KVCachePool::GetKVCachePoolStat() {
  auto & stat = kv_cache_pool_->kvc_pool_stat_;

#if 0
  {
    std::unique_lock lock{kv_cache_pool_->mut_};
    std::stringstream ss;
    ss << "kvc_blk_grps stat: ";
    for (auto & kvc_blk_grp : kv_cache_pool_->kvc_block_grps_) {
      if (kvc_blk_grp.stat.allocted) {
        ss << kvc_blk_grp.stat.num_used_blks << " "
          << kvc_blk_grp.stat.num_idle_blks << ", ";
      }
    }
    LOG(INFO) << ss.str();
  }
#endif

  return KVCachePoolStat {
    .num_idle_blk_grps = 
        stat.num_idle_blk_grps_.load(std::memory_order_relaxed),
    .num_allocated_blk_grps = 
        stat.num_allocated_blk_grps_.load(std::memory_order_relaxed)
  };
}

} // namespace colserve