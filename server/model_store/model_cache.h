#ifndef COLSERVE_MODEL_CACHE_H
#define COLSERVE_MODEL_CACHE_H

#include <server/config.h>
#include <server/schedule/job_queue.h>

#include <common/util.h>

#include <memory>
#include <mutex>
#include <vector>


namespace colserve {

class Model;
class InferModelStore;

/* [Note: Cache]
 *   we will have two level cache,
 *   1. warm cache: clasic model cache in infer server, 
 *      swap in/out model from/to the host memory to 
 *      over-subscribe the GPU memory
 * 
 *   2. cold cache: cache idle model in gpu to avoid some
 *      cold start
 */
class WarmModelCache {
 public:
  WarmModelCache() : cached_nbytes_(0) {};
  static std::unique_lock<std::mutex> ReserveCache(const std::string &name, size_t rank);
  static std::unique_lock<std::mutex> OrderedReserveCache(
      const std::string &name, size_t rank,
      const std::vector<std::shared_ptr<Job>> &jobs);
  static std::unique_lock<std::mutex> Lock() {
    if (Enable()) {
      return std::unique_lock{infer_model_cache_->mut_};
    } else {
      return std::unique_lock<std::mutex>{};
    }
  }
  
  static bool Enable() { return Config::max_warm_cache_nbytes != 0; }

  friend class InferModelStore;

 private:
  struct CacheItem {
    Model *model;
    bool cached;
    std::mutex mut;
  };

  static void Init() {
    infer_model_cache_ = std::make_unique<WarmModelCache>();
  }
  void MaybeAddCacheItem(const std::string &name, Model *model);
  void ReserveCacheInternal(const std::string &name, size_t rank,
                            std::unique_lock<std::mutex> &reserved_lock);

  static std::unique_ptr<WarmModelCache> infer_model_cache_;

  std::mutex mut_;
  std::condition_variable fifo_cv_;
  size_t cached_nbytes_;
  std::unordered_map<std::string, std::unique_ptr<CacheItem>> warm_cache_;

};

class ColdModelCache {
 private:
  static std::unique_ptr<ColdModelCache> cold_model_cache_;

  struct CacheItem {
    Model *model;
    std::vector<size_t> cached_groups_id;
    size_t cached_group_nbytes;
  };

  size_t current_cached_nbytes_;
  std::mutex mut_;
  std::unordered_map<std::string, std::unique_ptr<CacheItem>> cold_cache_;

 public:
  enum class ReservePolicy {
    kNotReserve = 0,
    kMaxCap = 1,
    kMaxMinDiff = 2,
  };

  static ReservePolicy reserve_policy_on_release;
  static ReservePolicy reserve_policy_on_adjust;

  using evict_list = std::vector<std::pair<std::string, std::vector<size_t>>>;
  using group_id_list = std::vector<size_t>;
  
  ColdModelCache(): current_cached_nbytes_(0) {}

  friend class InferModelStore;

  /**
   * Pushes a cache item into the model store.
   *
   * @param name The name of the cache item.
   * @param rank The rank of the cache item.
   * @param groups_nbytes The vector of group sizes in bytes.
   * @param total_nbytes The total size of the cache item in bytes.
   * @param source_model The loading model that causes pushing cache item.
   * @param lock The unique lock used to synchronize access to the infer model store.
   * @return A tuple containing the vector of group sizes, 
   *         a vector of pairs representing the names and group sizes,
   *         and a boolean indicating the success of the operation.
   */
  std::tuple<std::vector<size_t>, std::vector<std::pair<std::string, std::vector<size_t>>>, bool>
  PushCacheItem(const std::string& name, size_t rank, 
                std::vector<size_t> groups_nbytes, size_t total_nbytes, 
                std::unique_lock<std::mutex> &lock, Model *source_model);


  /**
   * @brief Pops a cache item from the infer model store.
   *
   * This function removes and returns a cache item identified by 
   * the given name and rank from the infer model store.
   *
   * @param name The name of the cache item to pop.
   * @param rank The rank of the cache item to pop.
   * @param lock A unique lock that ensures thread safety during the operation.
   * @return A pair containing the vector of sizes associated with the cache item 
   *         and a boolean indicating whether the pop operation was successful.
   */
  std::pair<std::vector<size_t>, bool> 
  PopCacheItem(const std::string& name, size_t rank, std::unique_lock<std::mutex> &lock);

  /**
   * Retrieves the list of models that are eligible for eviction from the infer model store.
   * 
   * @param capacity The desired capacity after eviction.
   * @param ignore_models 
   * @param lock A unique lock on the mutex.
   * @return The list of models that should be evicted.
   */
  evict_list GetEvictModels(long capacity, std::array<Model*, 2> ignore_models, 
                            std::unique_lock<std::mutex>& lock);


  std::unique_lock<std::mutex> Lock() {
    return std::unique_lock{mut_};
  }

  inline size_t GetCachedNbytes(std::unique_lock<std::mutex> &lock) {
    return current_cached_nbytes_;
  }

  inline size_t GetCachedNbytesUnsafe() {
    return current_cached_nbytes_;
  }

  inline size_t GetColdCacheReleasableMemoryMBUnsafe() {
    if (current_cached_nbytes_ > Config::cold_cache_min_capability_nbytes) {
      return sta::ByteToMB(current_cached_nbytes_ - Config::cold_cache_min_capability_nbytes);
    } else {
      return 0;
    }
  }

  double GetBufferMBUnsafe();

  double GetCacheSizeMBUnsafe();


  inline double GetColdCacheFreeMemoryMB(double free_memory_MB, std::unique_lock<std::mutex> &lock) {
    if (current_cached_nbytes_ > Config::cold_cache_min_capability_nbytes){
      free_memory_MB += sta::ByteToMB(current_cached_nbytes_ - Config::cold_cache_min_capability_nbytes);
    }
    // LOG(INFO) << "[ColdModelCache] FreeMemory " << free_memory_MB << "MB";
    return free_memory_MB;
  }

  double GetReleaseReserveMemoryMB(std::unique_lock<std::mutex> &lock);  
  double GetAdjustReserveMemoryMB(std::unique_lock<std::mutex> &lock);  

  static void Init() {
    cold_model_cache_ = std::make_unique<ColdModelCache>();
  }

  static ColdModelCache &Get() {
    CHECK(cold_model_cache_ != nullptr);
    return *cold_model_cache_;
  }
};



}

#endif