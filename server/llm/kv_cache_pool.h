#pragma once

#include <server/llm/llm_util.h>
#include <common/cuda_allocator.h>
#include <boost/python/dict.hpp>

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace colserve {

// LLM w/ mpool
//   LLM use mpool as kv-cache pool to adjust memory with training,
//   a group of kv-cache blocks is the smallest unit of memory allocation.
//   Model weight and activations will be managed be pytorch native 
//   memory managemetn

// KVCachePool w/ vLLM
//   vLLM use manage kv-cache by block idx. We should map the block idx
//   to the memory block of mpool


class KVCachePool {
 public:
  static void Init();
  static void MaybeSetKVCacheBlockNbytes(
    int64_t block_size, 
    int64_t num_layers, int64_t num_heads, int64_t head_size, 
    memory_byte_t block_nbytes    
  );
  static int GetNumGpuKVCacheBlocks();
  static PyObject* InitKVCache(
      int layer,
      const bp::list &shape,
      std::string dtype,
      int64_t item_size);
  static void PostInit();

  static void SetupFreeKVCacheBlockIndices(bp::list num_blocks);
  static uint64_t GetNumFreeBlocks();
  static void EnsureKVCacheBlock(uint64_t blk_idx);
  static void FreeKVCacheBlock(uint64_t blk_idx);
  static uint64_t AllocKVCacheBlock();
  // static void ReclaimKVCacheBlocks(
  //     bp::object free_block_indiecs, 
  //     bp::object free_blk_indices_not_allocted);

  KVCachePool() = default;

 private:
  static std::unique_ptr<KVCachePool> kv_cache_pool_;

  // a group of kv-cache blocks (cross all layers), 
  // in each layer, the kv-cache blocks consist of a page of 32MB
  struct KVCachePageGroup {
    std::array<uint64_t, 128> layer_kvc_blk_page_base_addr;
    struct Stat {
      bool allocted{false};
      uint64_t num_idle_blks{0};
      uint64_t num_used_blks{0};
      std::bitset<4096> blk_usage_bm;
    } stat;
  };


  std::pair<uint64_t, uint64_t> BlkIdxToPageGrpIdx(uint64_t blk_idx) {
    return {blk_idx / num_blocks_per_page_, 
            blk_idx % num_blocks_per_page_};
  }

  std::mutex mut_;
  int64_t block_size_{0}, num_layer_{0}, 
          num_heads_{0}, head_size_{0};
  memory_byte_t cache_block_nbytes_{0};
  uint64_t num_blocks_per_page_{0};
  uint64_t num_pages_per_layer_{0};

  std::array<
    std::unordered_map<
      uint64_t, 
      std::shared_ptr<sta::CUDAMemPool::PoolEntry>
    >, 128> layer_kvc_blk_pages;
  
  std::array<uint64_t, 128> kvc_space_base_addrs_{0};
  std::vector<KVCachePageGroup> kvc_page_groups_;

  std::set<uint64_t> free_blk_indices_,
                     free_blk_indices_not_allocted_;

  bool initialized_{false};
};

}