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

struct KVCachePoolStat {
  uint64_t num_idle_blk_grps;
  uint64_t num_allocated_blk_grps;
};

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
  static int64_t GetNumFreeLayerBlocks();
  static void UpdateNumFreeLayerBlocks(int64_t delta);
  static void UpdateNumFreeLayerBlocksByTrainBase(memory_byte_t train_base);
  static void FreeKVCacheBlock(const bp::list &blk_indices);
  static bp::list AllocKVCacheBlock(size_t n);
  static double GetKVCacheMemPageUtil();
  static KVCachePoolStat GetKVCachePoolStat();
  


  // static void ReclaimKVCacheBlocks(
  //     bp::object free_block_indiecs, 
  //     bp::object free_blk_indices_not_allocted);

  KVCachePool() = default;

  struct KVCacheBlockGroup {
    std::array<
      std::pair<uint64_t, uint64_t>, 128
    > layer_kvc_page_base_addr; // [k-cache-page, v-cache-page] x layer
    std::vector<uint64_t> block_indices;
    struct Stat {
      bool allocted{false};
      uint64_t num_idle_blks{0};
      uint64_t num_used_blks{0};
      std::bitset<4096> blk_usage_bm;
    } stat;
  };

 private:
  static std::unique_ptr<KVCachePool> kv_cache_pool_;
  static uint64_t cls_kv_cache_block_nbytes_;

  constexpr static uint64_t PhyPageFactor() {
    return 5;
  }
  constexpr static uint64_t KVCPageNybtes() {
    return sta::CUDAMemPool::PageNbytes() * PhyPageFactor();
  }

  // a group of kv-cache blocks (cross all layers), 
  // in each layer, the kv-cache blocks consist of a page of 32MB

  // struct KVCacheMemBlockExtraData {
  // };

  std::pair<uint64_t, uint64_t> BlkIdxToBlkGrpIdx(uint64_t blk_idx) {
    return {blk_idx / kvc_blk_grp_size_, 
            blk_idx % kvc_blk_grp_size_};
  }
  bool ReclaimKVCacheBlkGrpByBlkIdx(uint64_t blk_idx, 
                                    std::unique_lock<std::mutex> &lock);
  bool AllocKVCacheBlkGrp(std::unique_lock<std::mutex> &lock);
  void ReclaimAllKVCache(std::unique_lock<std::mutex> &lock);
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

  // deprecated
  std::unordered_map<uint64_t, std::shared_ptr<sta::CUDAMemPool::PoolEntry>> kv_cache_blocks_;
  uint64_t kvc_blk_grp_size_{0};
  uint64_t num_kvc_blk_grp_per_layer_{0};

  struct KVCachePoolStatInternal {
    // a blk_grp is used when any block in it is used
    std::atomic<uint64_t> num_idle_blk_grps_{0};
    std::atomic<uint64_t> num_allocated_blk_grps_{0};

    std::atomic<uint64_t> num_used_blks_{0};
  } kvc_pool_stat_;

  bool initialized_{false};
};

}