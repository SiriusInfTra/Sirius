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
  static int64_t GetBlockNbytes();
  static int64_t GetNumLayers(); 
  static PyObject* CreateKVCache(
      std::string dtype,
      int64_t item_size);
  static std::vector<int64_t> GetKVCacheShape();
  static std::vector<int64_t> GetKVCacheStride();
  static std::pair<int64_t, int64_t> GetNumLayerBlockInfo();
  static void UpdateNumFreeLayerBlocks(int64_t delta);
  static void UpdateNumFreeLayerBlocksByTrainBase(memory_byte_t train_base);
  static void FreeKVCachePage(size_t page_nbytes, const bp::list &page_indices);
  static bp::list AllocKVCachePage(size_t page_nbytes, size_t n);
  static double GetKVCachePoolkUtil();
  
  KVCachePool() = default;

 private:
  static std::unique_ptr<KVCachePool> kv_cache_pool_;
  static uint64_t cls_kv_cache_block_nbytes_;

  bool ReclaimKVCacheBlkGrpByBlkIdx(uint64_t blk_idx, 
                                    std::unique_lock<std::mutex> &lock);
  uint64_t MaybeAdjustTrain(uint64_t num_required_pages, 
                            uint64_t min_num_required_pages,
                            std::unique_lock<std::mutex> &lock);
  void ReclaimMemToTrain(std::unique_lock<std::mutex> &kvc_pool_lock);

  std::mutex mut_;
  int64_t block_size_{0}, num_layer_{0}, 
          num_heads_{0}, head_size_{0};
  memory_byte_t cache_block_nbytes_{0};
  int64_t num_kvc_blks_{-1};
  bool has_set_block_shape_{false};

  int64_t num_free_layer_blks_{-1},
          num_used_layer_blks_{-1};

  std::unordered_map<
    uint64_t, 
    std::shared_ptr<sta::CUDAMemPool::PoolEntry>
  > kv_cache_pages_;

  bool initialized_{false};
};

}