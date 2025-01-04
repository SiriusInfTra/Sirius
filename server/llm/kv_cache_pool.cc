#include <server/llm/kv_cache_pool.h>
#include <boost/range/irange.hpp>

#include <c10/core/TensorImpl.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>

namespace colserve {

std::unique_ptr<KVCachePool> KVCachePool::kv_cache_pool_ = nullptr;

void KVCachePool::Init() {
  CHECK(sta::CUDAMemPool::IsEnable());
  kv_cache_pool_ = std::make_unique<KVCachePool>();
}

void KVCachePool::CheckKVCacheBlockNbytes(
    int64_t block_size,
    int64_t num_layers, 
    int64_t num_heads, 
    int64_t head_size, 
    memory_byte_t block_nbytes) {
  CHECK(kv_cache_pool_ != nullptr);
  CHECK(32_MB % block_nbytes == 0) 
      << "block_nbytes " << block_nbytes 
      << " is not a factor of 32_MB";
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
  CHECK(kv_cache_pool_ != nullptr);
  auto ret = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
      ->PoolNbytes() / kv_cache_pool_->cache_block_nbytes_;
  return ret;
}

PyObject* KVCachePool::InitKVCache(
    int layer, const bp::list &shape_, std::string dtype, int64_t itemsize) {
  std::vector<int64_t> shape;
  for (auto i : boost::irange(bp::len(shape_))) {
    shape.push_back(bp::extract<int64_t>(shape_[i]));
  }

  CHECK(kv_cache_pool_ != nullptr);
  CHECK(layer <= kv_cache_pool_->layer_kv_cache_blocks.size());
  // at::ten
  // torch::from_blob()
  memory_byte_t total_nbytes_per_layer = 1;
  for (auto dim : shape) {
    // block_nbytes_per_layer *= dim;
    total_nbytes_per_layer *= dim;
  }
  CHECK_EQ(shape.size(), 3) << "shape " << shape;
  memory_byte_t block_nbytes_per_layer = shape[2];

  CHECK_EQ(block_nbytes_per_layer, 
    kv_cache_pool_->num_heads_ 
    * kv_cache_pool_->block_size_ 
    * kv_cache_pool_->head_size_
  );
  block_nbytes_per_layer *= itemsize;
  total_nbytes_per_layer *= itemsize;
  
  CHECK(total_nbytes_per_layer % 32_MB == 0);
  auto num_blocks_per_page = 32_MB / block_nbytes_per_layer;
  auto num_pages_per_layer = total_nbytes_per_layer / 32_MB;
  DLOG(INFO) << "[InitKVCache] layer " << layer
            << " block_nbytes_per_layer " << sta::PrintByte(block_nbytes_per_layer)
            << " total_nbytes_per_layer " << sta::PrintByte(total_nbytes_per_layer)
            << " num_blocks_per_page " << num_blocks_per_page
            << " num_pages_per_layer " << num_pages_per_layer;

  std::unique_lock lock{kv_cache_pool_->mut_};
  std::shared_ptr<sta::CUDAMemPool::PoolEntry> last_entry = nullptr, 
                                               first_entry = nullptr;

  // ensure the kv_cache_block for a layer is continuous
  // concurrent allocation must be avoided
  for (auto i : boost::irange(num_pages_per_layer)) {
    auto entry = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
        ->Alloc(32_MB, sta::MemType::kInfer, false);
    CHECK((reinterpret_cast<size_t>(entry->addr) & (32_MB - 1)) == 0);
    auto& layer_kv_cache_blocks = 
        kv_cache_pool_->layer_kv_cache_blocks[layer][reinterpret_cast<uint64_t>(entry->addr)];
    layer_kv_cache_blocks.entry = entry;
    layer_kv_cache_blocks.stat = KVCacheBlock::Stat {
      .allocted = true,
      .num_free_blks = num_blocks_per_page,
      .num_used_blks = 0
    };
    if (last_entry != nullptr) {
      CHECK_EQ(reinterpret_cast<size_t>(last_entry->addr) + 32_MB, 
               reinterpret_cast<size_t>(entry->addr))
          << "kv_cache_block for a layer must be continuous";
    }
    last_entry = entry;
    if (first_entry == nullptr) {
      first_entry = entry;
    }
  }

  try {
    auto tensor_opt = at::TensorOptions()
        .device(at::kCUDA, sta::DeviceManager::GetCurrentDevice());
    if (dtype == "torch.float32") {
      tensor_opt = tensor_opt.dtype(at::kFloat);
    } else if (dtype == "torch.float16") {
      tensor_opt = tensor_opt.dtype(at::kHalf);
    } else {
      LOG(FATAL) << "unsupported dtype " << dtype;
    }

    auto tensor = at::from_blob(first_entry->addr, shape, tensor_opt);
    return torch::autograd::utils::wrap(tensor);
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    LOG(FATAL) << "InitKVCache failed: " << e.what();
  }
}

}