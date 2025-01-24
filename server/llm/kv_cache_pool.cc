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
  
  /* Note: kv cache shape
   *   vLLM kv cache shape can be treated as: [#layer, 2, #blk, blk_size * #head * head_size]
   *   (actually, one tensor for one layer)
   *   with following layout:
   *      <----- k-cache ---->|<---- v-cache --->
   *      |K_1|K_2|K_3|K_4|...|V_1|V_2|V_3|V_4|...  (layer i)
   *        ↳ k-cache of blk_size tokens
   *        
   *   we organize the kv cache as a single tensor, 
   *   shape: [2, #layer * #blk, blk_size * #head * head_size]
   *   layout:
   *      |K_1,V_1|K_2,V_2|K_3,V_3|K_4,V_4|...
   */

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
  kv_cache_pool_->has_set_block_shape_ = true;
}

int KVCachePool::GetNumGpuKVCacheBlocks() {
  // CHECK(KVCachePool::cls_kv_cache_block_nbytes_ != 0);
  if (!Config::use_shared_tensor) {
    return Config::cuda_memory_pool_gb * 1_GB / KVCachePool::cls_kv_cache_block_nbytes_;
  }

  CHECK(kv_cache_pool_ != nullptr);
  if (kv_cache_pool_->num_kvc_blks_ != -1) {
    return kv_cache_pool_->num_kvc_blks_;
  }

  size_t cache_blk_nbytes = 
    2     /* float16 */
    * 2   /* k/v     */
    * kv_cache_pool_->block_size_
    * kv_cache_pool_->num_heads_
    * kv_cache_pool_->head_size_
    * kv_cache_pool_->num_layer_;
  CHECK_EQ(cache_blk_nbytes, kv_cache_pool_->cache_block_nbytes_);

  auto ret = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
      ->PoolNbytes() / KVCachePool::cls_kv_cache_block_nbytes_;

  LOG(INFO) << "[GetNumGpuKVCacheBlocks] "
            << "cache_blk_nbytes " << sta::PrintByte(cache_blk_nbytes)
            << " #blk " << ret;
  kv_cache_pool_->num_kvc_blks_ = ret;


  return ret;
}

int64_t KVCachePool::GetBlockNbytes() {
  CHECK(kv_cache_pool_ != nullptr);
  CHECK(kv_cache_pool_->has_set_block_shape_);
  return kv_cache_pool_->cache_block_nbytes_;
}

int64_t KVCachePool::GetNumLayers() {
  CHECK(kv_cache_pool_ != nullptr);
  CHECK(kv_cache_pool_->has_set_block_shape_);
  return kv_cache_pool_->num_layer_;
}

PyObject* KVCachePool::CreateKVCache(std::string dtype, int64_t itemsize) {
  CHECK(kv_cache_pool_ != nullptr);
  CHECK(kv_cache_pool_->has_set_block_shape_)
      << "kv cache shape must be set before create kv cache";
  CHECK_EQ(itemsize, 2) << "only support float16/half";

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
    // auto 
    auto tensor = at::from_blob(
        sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice()
      )->GetBasePtr(sta::MemType::kInfer), 
      kv_cache_pool_->GetKVCacheShape(), 
      kv_cache_pool_->GetKVCacheStride(),
      tensor_opt);
    return torch::autograd::utils::wrap(tensor);
  } catch (const std::exception& e) {
    LOG(FATAL) << "InitKVCache failed: " << e.what();
  }
}

std::vector<int64_t> KVCachePool::GetKVCacheShape() {
  CHECK(kv_cache_pool_ != nullptr);
  CHECK(kv_cache_pool_->has_set_block_shape_);
  
  auto & kvc_pool = kv_cache_pool_;
  std::vector<int64_t> ret{
    2,
    kvc_pool->num_layer_ * kvc_pool->GetNumGpuKVCacheBlocks(),
    kvc_pool->block_size_ * kvc_pool->num_heads_ * kvc_pool->head_size_,
  };
  return ret;
}

std::vector<int64_t> KVCachePool::GetKVCacheStride() {
  CHECK(kv_cache_pool_ != nullptr);
  CHECK(kv_cache_pool_->has_set_block_shape_);
  
  auto & kvc_pool = kv_cache_pool_;
  std::vector<int64_t> ret{
    kvc_pool->block_size_ * kvc_pool->num_heads_ * kvc_pool->head_size_,
    kvc_pool->block_size_ * kvc_pool->num_heads_ * kvc_pool->head_size_ * 2,
    1
  };
  return ret;
}

std::pair<int64_t, int64_t> KVCachePool::GetNumLayerBlockInfo() {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};
  if (kv_cache_pool_->num_free_layer_blks_ == -1) {
    CHECK(kv_cache_pool_->num_kvc_blks_ > 0);
    CHECK(kv_cache_pool_->has_set_block_shape_);
    kv_cache_pool_->num_free_layer_blks_ = 
        kv_cache_pool_->num_kvc_blks_ * kv_cache_pool_->num_layer_;
    kv_cache_pool_->num_used_layer_blks_ = 0;

    auto num_vllm_blk_env = std::getenv("COLSYS_VLLM_NUM_LAYER_BLOCK");
    if (num_vllm_blk_env != nullptr) {
      auto num_vllm_blk = std::stoll(num_vllm_blk_env);
      CHECK_GT(num_vllm_blk, 0);
      CHECK_LE(num_vllm_blk, kv_cache_pool_->num_free_layer_blks_);
      kv_cache_pool_->num_free_layer_blks_ = num_vllm_blk;
      LOG(INFO) << "[GetNumFreeLayerBlocks] COLSYS_VLLM_NUM_LAYER_BLOCK env: "
                << " num_free_layer_blks " 
                << kv_cache_pool_->num_free_layer_blks_;
    }
  }
  return {kv_cache_pool_->num_free_layer_blks_, 
          kv_cache_pool_->num_used_layer_blks_};
}

void KVCachePool::UpdateNumFreeLayerBlocks(int64_t delta) {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};
  CHECK_GE(kv_cache_pool_->num_free_layer_blks_, 0);

  kv_cache_pool_->num_free_layer_blks_ += delta;
  kv_cache_pool_->num_used_layer_blks_ -= delta;

  CHECK_GE(kv_cache_pool_->num_free_layer_blks_, 0);
}

void KVCachePool::UpdateNumFreeLayerBlocksByTrainBase(memory_byte_t train_base) {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};
  CHECK(kv_cache_pool_->has_set_block_shape_);
  CHECK_GE(kv_cache_pool_->num_free_layer_blks_, 0);
  CHECK_EQ(kv_cache_pool_->num_used_layer_blks_, 0);

  int64_t num_free_blks = 
      kv_cache_pool_->GetNumGpuKVCacheBlocks()
      - (train_base 
         + Config::train_memory_over_predict_mb * 1_MB
        ) / kv_cache_pool_->cache_block_nbytes_; 
  CHECK_GE(num_free_blks, 0);

  auto old_num_free_layer_blks = kv_cache_pool_->num_free_layer_blks_;
  kv_cache_pool_->num_free_layer_blks_ = num_free_blks * kv_cache_pool_->num_layer_;

  LOG(INFO) << "[UpdateNumFreeLayerBlocksByTrainBase] "
            << " num_free_layer_blks " << old_num_free_layer_blks
            << " -> " << kv_cache_pool_->num_free_layer_blks_;
}

void KVCachePool::FreeKVCachePage(size_t page_nbytes, 
                                  const bp::list &page_indices) {
  CHECK(kv_cache_pool_ != nullptr);
  // size_t page_nbytes = 5 * sta::CUDAMemPool::PageNbytes();
  std::unique_lock lock{kv_cache_pool_->mut_};
  for (auto k : boost::irange(bp::len(page_indices))) {
    auto pg_idx = bp::extract<int64_t>(page_indices[k]);
    auto iter = kv_cache_pool_->kv_cache_pages_.find(pg_idx);
    CHECK(iter != kv_cache_pool_->kv_cache_pages_.end());
    kv_cache_pool_->kv_cache_pages_.erase(iter);
  }
  
  if (!ctrl::Controller::Get()->IsTrainIdle() &&
    sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice()
      )->NumFreePages() * sta::CUDAMemPool::PageNbytes() >= 6_GB) {
    kv_cache_pool_->ReclaimMemToTrain(lock);
  }
}

bp::list KVCachePool::AllocKVCachePage(size_t page_nbytes, size_t n) {
  CHECK(kv_cache_pool_ != nullptr);
  // NOTE: nbytes = 160MB * n
  // size_t page_nbytes = 160_MB;
  auto *mem_pool = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice());
  std::unique_lock lock{kv_cache_pool_->mut_};
  int64_t train_memory = std::max(
    static_cast<size_t>(TrainAdjuster::PredictTrainMemUsageMB(
      sta::DeviceManager::GetCurrentDevice(), true) * 1_MB),
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
    CHECK(kv_cache_pool_->kv_cache_pages_.insert(std::make_pair(blk, block)).second);
    ret.append(blk);
  }
  
  return ret;
}

uint64_t KVCachePool::MaybeAdjustTrain(
    uint64_t num_required_pages,
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

double KVCachePool::GetKVCachePoolkUtil() {
  if (kv_cache_pool_ == nullptr 
      || !kv_cache_pool_->has_set_block_shape_) {
    return std::nan("");
  }

  auto infer_mem = sta::CUDAMemPool::Get(
      sta::DeviceManager::GetCurrentDevice()
    )->InferMemUsage();

  return 1.0 * kv_cache_pool_->num_used_layer_blks_ 
      * kv_cache_pool_->cache_block_nbytes_ 
      / kv_cache_pool_->num_layer_ / infer_mem; 
}


} // namespace colserve