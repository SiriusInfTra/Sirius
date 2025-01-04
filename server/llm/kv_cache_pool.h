#pragma once

#include <server/llm/llm_util.h>
#include <common/cuda_allocator.h>

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace colserve {

// LLM w/ mpool
//   LLM use mpool as kv-cache pool to adjust memory with training,
//   a group of kv-cache blocks is the smallest unit of memory allocation.
//   Model weight and activations will be managed be pytorch native 
//   memory managemetn


class KVCachePool {
 public:
  static void Init();
  static void CheckKVCacheBlockNbytes(
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

  KVCachePool() = default;

 private:
  static std::unique_ptr<KVCachePool> kv_cache_pool_;

  struct KVCacheBlock {
    std::shared_ptr<sta::CUDAMemPool::PoolEntry> entry;
    struct Stat {
      bool allocted{false};
      uint64_t num_free_blks{0};
      uint64_t num_used_blks{0};
    } stat;
  };

  std::mutex mut_;
  int64_t block_size_, num_layer_, num_heads_, head_size_;
  memory_byte_t cache_block_nbytes_{0};
  std::array<
    std::unordered_map<uint64_t, KVCacheBlock>, 128
  > layer_kv_cache_blocks;
};

}