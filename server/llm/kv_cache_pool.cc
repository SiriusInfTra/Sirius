#include <server/llm/kv_cache_pool.h>
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
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};

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
  CHECK(layer <= kv_cache_pool_->kvc_blk_pages_.size());
  // at::ten
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
  
  CHECK(total_nbytes_per_layer % 32_MB == 0);
  auto num_blocks_per_page = 
      32_MB / block_nbytes_per_layer 
      * 2 /* k-cache in a page, v-cache in other page */;
  auto num_pages_per_layer = total_nbytes_per_layer / 32_MB;

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
  
  DLOG(INFO) << "[InitKVCache] layer " << layer
            << " block_nbytes_per_layer " << sta::PrintByte(block_nbytes_per_layer)
            << " total_nbytes_per_layer " << sta::PrintByte(total_nbytes_per_layer)
            << " kvc_blk_grp_size " << kv_cache_pool_->kvc_blk_grp_size_
            << " num_pages_per_layer " << num_pages_per_layer
            << " num_kvc_blk_grp_per_layer " 
            << kv_cache_pool_->num_kvc_blk_grp_per_layer_;

  std::unique_lock lock{kv_cache_pool_->mut_};
  std::shared_ptr<sta::CUDAMemPool::PoolEntry> last_entry = nullptr, 
                                               first_entry = nullptr;

  // ensure the kv_cache_block for a layer is continuous
  // concurrent allocation must be avoided
  for (auto i : boost::irange(num_pages_per_layer)) {
    auto entry = sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
        ->Alloc(32_MB, sta::MemType::kInfer, false);
    CHECK((reinterpret_cast<size_t>(entry->addr) & (32_MB - 1)) == 0);
    kv_cache_pool_->kvc_blk_pages_[layer][
        reinterpret_cast<uint64_t>(entry->addr)
      ] = entry;
    // auto& layer_kv_cache_blocks = 
    //     kv_cache_pool_->layer_kv_cache_blocks[layer][reinterpret_cast<uint64_t>(entry->addr)];
    // layer_kv_cache_blocks.entry = entry;
    // layer_kv_cache_blocks.stat = KVCacheBlock::Stat {
    //   .allocted = true,
    //   .num_free_blks = num_blocks_per_page,
    //   .num_used_blks = 0
    // };
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

  CHECK(first_entry != nullptr);
  kv_cache_pool_->kvc_space_base_addrs_[layer] = 
    reinterpret_cast<size_t>(first_entry->addr);

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
  } catch (const std::exception& e) {
    LOG(FATAL) << "InitKVCache failed: " << e.what();
  }
}

void KVCachePool::PostInit() {
  CHECK(kv_cache_pool_ != nullptr);
  LOG(INFO) << "[KVCachePool] PostInit";
  std::unique_lock lock{kv_cache_pool_->mut_};

  CHECK(kv_cache_pool_->cache_block_nbytes_ != 0);
  CHECK(kv_cache_pool_->block_size_ != 0);
  CHECK(kv_cache_pool_->num_layer_ != 0);
  CHECK(kv_cache_pool_->num_heads_ != 0);
  CHECK(kv_cache_pool_->head_size_ != 0);
  CHECK(kv_cache_pool_->kvc_blk_grp_size_ != 0);
  CHECK(kv_cache_pool_->num_kvc_blk_grp_per_layer_ != 0);
  for (auto i : boost::irange(kv_cache_pool_->num_layer_)) {
    CHECK(kv_cache_pool_->kvc_space_base_addrs_[i] != 0);
  }

  LOG(INFO) << "[KVCachePool] PostInit, "
            << "cache_block_nbytes " 
            << sta::PrintByte(kv_cache_pool_->cache_block_nbytes_)
            << " block_size " << kv_cache_pool_->block_size_
            << " num_layer " << kv_cache_pool_->num_layer_
            << " num_heads " << kv_cache_pool_->num_heads_
            << " head_size " << kv_cache_pool_->head_size_
            << " kvc_blk_grp_size " << kv_cache_pool_->kvc_blk_grp_size_
            << " num_kvc_blk_grp_per_layer " 
            << kv_cache_pool_->num_kvc_blk_grp_per_layer_;

  auto & kvc_block_grps = kv_cache_pool_->kvc_block_grps_;
  kvc_block_grps.resize(kv_cache_pool_->num_kvc_blk_grp_per_layer_);
  
  for (auto i : boost::irange(kvc_block_grps.size())) {
    kvc_block_grps[i].stat = KVCacheBlockGroup::Stat{
      .allocted = true,
      .num_idle_blks = kv_cache_pool_->kvc_blk_grp_size_,
      .num_used_blks = 0
    };
    kvc_block_grps[i].stat.blk_usage_bm.reset();
    kvc_block_grps[i].stat.blk_usage_bm.flip();
    for (auto j : boost::irange(kv_cache_pool_->kvc_blk_grp_size_)) {
      kvc_block_grps[i].stat.blk_usage_bm[j] = false;
    }
    auto block_indices = boost::irange(
        i * kv_cache_pool_->kvc_blk_grp_size_, 
        (i + 1) * kv_cache_pool_->kvc_blk_grp_size_);
    kvc_block_grps[i].block_indices.assign(
        block_indices.begin(), block_indices.end());
  }

  for (auto layer_idx : boost::irange(kv_cache_pool_->num_layer_)) {
    auto & layer_kvc_pages = kv_cache_pool_->kvc_blk_pages_[layer_idx];
    auto & layer_kvc_space_base_addr = kv_cache_pool_->kvc_space_base_addrs_[layer_idx];

    CHECK(layer_kvc_pages.size() % 2 == 0);
    auto num_k_cache_pages = layer_kvc_pages.size() / 2;

    for (auto & [addr, entry] : layer_kvc_pages) {
      auto page_idx = 
          (reinterpret_cast<size_t>(addr) - layer_kvc_space_base_addr) / 32_MB;
      if (page_idx < num_k_cache_pages) {
        CHECK(page_idx < kvc_block_grps.size());
        auto & kvc_blk_grp = kvc_block_grps[page_idx];
        kvc_blk_grp.layer_kvc_page_base_addr[layer_idx].first = reinterpret_cast<size_t>(addr);
      } else {
        CHECK(page_idx - num_k_cache_pages < kvc_block_grps.size());
        auto & kvc_blk_grp = kvc_block_grps[page_idx - num_k_cache_pages];
        kvc_blk_grp.layer_kvc_page_base_addr[layer_idx].second = reinterpret_cast<size_t>(addr);
      }
    }
  }

  LOG(INFO) << "total page_groups " << kvc_block_grps.size()
            << " | kv_cache_page_group[0]: "
            << kvc_block_grps[0]
            << " | kv_cache_page_group[-1]: "
            << kvc_block_grps.back();

  auto & kvc_pool_stat = kv_cache_pool_->kvc_pool_stat_;
  kvc_pool_stat.num_idle_blk_grps_ = kvc_block_grps.size();
  kvc_pool_stat.num_allocated_blk_grps_ = kvc_block_grps.size();
  kvc_pool_stat.num_used_blks_ = 0;
  kv_cache_pool_->initialized_ = true;

  /////////////////////////////////////
  // reclaim allocated free blocks
  auto blk_idx_it = kv_cache_pool_->free_blk_indices_.begin();
  while (blk_idx_it != kv_cache_pool_->free_blk_indices_.end()) {
    auto blk_idx = *blk_idx_it;
    bool succ = kv_cache_pool_
        ->ReclaimKVCacheBlkGrpByBlkIdxWithoutLock(blk_idx);
    if (succ) {
      blk_idx_it = kv_cache_pool_->free_blk_indices_.lower_bound(blk_idx);
    } else {
      blk_idx_it++;
    }
  }
}

void KVCachePool::SetupFreeKVCacheBlockIndices(bp::list block_ids) {
  CHECK(kv_cache_pool_ != nullptr);
  std::vector<uint64_t> blk_ids_vec;
  for (auto i : boost::irange(bp::len(block_ids))) {
    CHECK_EQ(i, bp::extract<uint64_t>(block_ids[i]));
    blk_ids_vec.push_back(bp::extract<uint64_t>(block_ids[i]));
  } 
  CHECK(GetNumGpuKVCacheBlocks() == blk_ids_vec.size());

  std::unique_lock lock{kv_cache_pool_->mut_};
  kv_cache_pool_->free_blk_indices_.insert(blk_ids_vec.begin(), blk_ids_vec.end());
  kv_cache_pool_->free_blk_indices_not_allocted_.clear();

  LOG(INFO) << "free_blk_indices_ " << kv_cache_pool_->free_blk_indices_.size();
}

uint64_t KVCachePool::GetNumFreeBlocks() {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};
  return kv_cache_pool_->free_blk_indices_.size()
      + kv_cache_pool_->free_blk_indices_not_allocted_.size();
}

void KVCachePool::EnsureKVCacheBlock(uint64_t blk_idx) {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};

  auto & kvc_page_grps = kv_cache_pool_->kvc_block_grps_;

  auto [pg_grp_idx, blk_idx_in_page] = kv_cache_pool_->BlkIdxToBlkGrpIdx(blk_idx);
  CHECK(pg_grp_idx < kvc_page_grps.size());
  if (!kvc_page_grps[pg_grp_idx].stat.allocted) {
    LOG(FATAL) << "KVCachePageGroup " << pg_grp_idx << " is not allocted";
  }
}

void KVCachePool::FreeKVCacheBlock(uint64_t blk_idx) {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};

  CHECK(kv_cache_pool_->free_blk_indices_.find(blk_idx) == 
        kv_cache_pool_->free_blk_indices_.end());
  kv_cache_pool_->free_blk_indices_.insert(blk_idx);

  auto & kvc_page_grps = kv_cache_pool_->kvc_block_grps_;
  auto [pg_grp_idx, blk_idx_in_page] = kv_cache_pool_->BlkIdxToBlkGrpIdx(blk_idx);
  CHECK(pg_grp_idx < kvc_page_grps.size());

  if (!kvc_page_grps[pg_grp_idx].stat.allocted) {
    LOG(FATAL) << "KVCachePageGroup " << pg_grp_idx << " is not allocted";
  }

  auto & kvc_page_grp = kvc_page_grps[pg_grp_idx];
  if (kvc_page_grp.stat.blk_usage_bm[blk_idx_in_page]) {
    kvc_page_grp.stat.blk_usage_bm[blk_idx_in_page] = false;
    kvc_page_grp.stat.num_idle_blks++;
    kvc_page_grp.stat.num_used_blks--;
  } else {
    LOG(FATAL) << "KVCacheBlock " << blk_idx << " is not used";
  }

  auto & kvc_pool_stat = kv_cache_pool_->kvc_pool_stat_;
  if (kvc_page_grp.stat.num_used_blks == 0) {
    auto num_idle = kvc_pool_stat.num_idle_blk_grps_.fetch_add(
        1, std::memory_order_relaxed);
    if (num_idle >= 1) {
      kv_cache_pool_->ReclaimKVCacheBlkGrpByBlkIdxWithoutLock(blk_idx);
    }
  }
}

uint64_t KVCachePool::AllocKVCacheBlock() {
  CHECK(kv_cache_pool_ != nullptr);
  std::unique_lock lock{kv_cache_pool_->mut_};

  bool new_blk_grp = false;
  if (kv_cache_pool_->free_blk_indices_.empty()) {
    auto succ = kv_cache_pool_->AllocKVCacheBlkGrpWithoutLock();
    CHECK(succ) << "OOM: no free block";
    new_blk_grp = true;
  }

  auto blk_idx = *kv_cache_pool_->free_blk_indices_.begin();
  kv_cache_pool_->free_blk_indices_.erase(blk_idx);

  auto & kvc_page_grps = kv_cache_pool_->kvc_block_grps_;
  auto [pg_grp_idx, blk_idx_in_page] = kv_cache_pool_->BlkIdxToBlkGrpIdx(blk_idx);
  CHECK(pg_grp_idx < kvc_page_grps.size());

  auto & kvc_page_grp = kvc_page_grps[pg_grp_idx];

  if (!kvc_page_grp.stat.allocted) {
    LOG(FATAL) << "KVCachePageGroup " << pg_grp_idx << " is not allocted";
  }
  if (new_blk_grp) CHECK(kvc_page_grp.stat.num_used_blks == 0);

  if (kvc_page_grp.stat.num_used_blks == 0) {
    kv_cache_pool_->kvc_pool_stat_.num_idle_blk_grps_.fetch_sub(
        1, std::memory_order_relaxed);
  }
  if (!kvc_page_grp.stat.blk_usage_bm[blk_idx_in_page]) {
    kvc_page_grp.stat.blk_usage_bm[blk_idx_in_page] = true;
    kvc_page_grp.stat.num_idle_blks--;
    kvc_page_grp.stat.num_used_blks++;
  } else {
    LOG(FATAL) << "KVCacheBlock " << blk_idx << " is already used";
  }

  return blk_idx;
}

bool KVCachePool::ReclaimKVCacheBlkGrpByBlkIdxWithoutLock(uint64_t blk_idx) {
  auto [pg_grp_idx, blk_idx_in_page] = BlkIdxToBlkGrpIdx(blk_idx);
  CHECK(pg_grp_idx < kvc_block_grps_.size());
  CHECK(kvc_block_grps_[pg_grp_idx].stat.allocted) 
      << "KVCachePageGroup " << pg_grp_idx 
      << " (blk_idx "<< blk_idx << ") is not allocted " 
      << kvc_block_grps_[pg_grp_idx].stat.allocted
      << " | stat: num_idle_blks " << kvc_block_grps_[pg_grp_idx].stat.num_idle_blks
      << " num_used_blks " << kvc_block_grps_[pg_grp_idx].stat.num_used_blks;
  
  if (kvc_block_grps_[pg_grp_idx].stat.num_used_blks > 0) {
    return false;
  }

  // maintain the free block index
  auto & block_indices = kvc_block_grps_[pg_grp_idx].block_indices;
  for (auto blk_idx : block_indices) {
    auto it = free_blk_indices_.find(blk_idx);
    CHECK(it != free_blk_indices_.end());
    free_blk_indices_.erase(it);
    auto [_, succ] = free_blk_indices_not_allocted_.insert(blk_idx);
    CHECK(succ);
  }

  // unmap pages
  for (auto l : boost::irange(num_layer_)) {
    auto kvc_page = {
      kvc_block_grps_[pg_grp_idx].layer_kvc_page_base_addr[l].first,
      kvc_block_grps_[pg_grp_idx].layer_kvc_page_base_addr[l].second
    };
    for (auto addr : kvc_page) {
      auto it = kvc_blk_pages_[l].find(addr);
      CHECK(it != kvc_blk_pages_[l].end());
      sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
          ->Unmap(it->second);
    }
  }

  LOG(INFO) << "set " << pg_grp_idx << " " 
            << free_blk_indices_.size() << " " 
            << free_blk_indices_not_allocted_.size();
  kvc_block_grps_[pg_grp_idx].stat.allocted = false;
  kvc_pool_stat_.num_idle_blk_grps_.fetch_sub(
      1, std::memory_order_relaxed);
  kvc_pool_stat_.num_allocated_blk_grps_.fetch_sub(
      1, std::memory_order_relaxed);
  return true;
}

bool KVCachePool::AllocKVCacheBlkGrpWithoutLock() {
  auto it = free_blk_indices_not_allocted_.begin();
  if (it == free_blk_indices_not_allocted_.end()) {
    return false;
  }

  auto blk_idx = *it;
  auto [pg_grp_idx, blk_idx_in_page] = BlkIdxToBlkGrpIdx(blk_idx);
  CHECK(pg_grp_idx < kvc_block_grps_.size());
  CHECK(!kvc_block_grps_[pg_grp_idx].stat.allocted) 
      << "KVCachePageGroup " << pg_grp_idx << " is already allocted";

  // mapping pages
  for (auto l : boost::irange(num_layer_)) {
    auto kvc_page = {
      kvc_block_grps_[pg_grp_idx].layer_kvc_page_base_addr[l].first,
      kvc_block_grps_[pg_grp_idx].layer_kvc_page_base_addr[l].second
    };
    for (auto addr : kvc_page) {
      auto it = kvc_blk_pages_[l].find(addr);
      CHECK(it != kvc_blk_pages_[l].end());
      sta::CUDAMemPool::Get(sta::DeviceManager::GetCurrentDevice())
          ->Map(it->second);
    }
  }

  // maintain the free block indices
  auto & block_indices = kvc_block_grps_[pg_grp_idx].block_indices;
  for (auto blk_idx : block_indices) {
    auto it = free_blk_indices_not_allocted_.find(blk_idx);
    CHECK(it != free_blk_indices_not_allocted_.end());
    free_blk_indices_not_allocted_.erase(it);
    auto [_, succ] = free_blk_indices_.insert(blk_idx);
    CHECK(succ);
  }

  kvc_block_grps_[pg_grp_idx].stat.allocted = true;
  kvc_pool_stat_.num_idle_blk_grps_.fetch_add(
      1, std::memory_order_relaxed);
  kvc_pool_stat_.num_allocated_blk_grps_.fetch_add(
      1, std::memory_order_relaxed);
  return true;
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