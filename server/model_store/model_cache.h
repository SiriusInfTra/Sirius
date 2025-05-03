#ifndef COLSERVE_MODEL_CACHE_H
#define COLSERVE_MODEL_CACHE_H

#include <server/logging_as_glog.h>
#include <server/config.h>
#include <server/schedule/job_queue.h>

#include <common/util.h>

#include <cstddef>
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
 * 
 * [Note: Cache & Lock]
 *   we want to maintain following properties:
 *   1. one model can only be in warm cache or cold cache.
 *   2. if a model is used, it should be in warm cache, and will
 *      not be evicted util it is not used.
 *   3. if a model is evicted from warm cache, it should first
 *      be put into cold cache.
 * 
 *   Further, we need to lock the protect cache data structure when 
 *   model are add/delete from the cache. We need to avoid dead lock
 *   during maintaining the cache.
 *   There are involved locks:
 *   1. model lock: protect the model itself
 *   2. warm cache lock: protect the warm cache
 *      per-model warm cache lock: to maintain the model in warm cache
 *   3. cold cache lock: protect the cold cache
 * 
 *   Lock convention: warm cache -> cold cache -> model
 *  
 */
class WarmModelCache {
 public:
  static bool Enable() { return Config::max_warm_cache_nbytes != 0; }
  static WarmModelCache* Get(int device_id);


  WarmModelCache(int device_id) : cached_nbytes_{0}, device_id_{device_id} {};
  std::unique_lock<std::mutex> ReserveCache(const std::string &name, size_t rank);
  std::unique_lock<std::mutex> OrderedReserveCache(
      const std::string &name, size_t rank,
      const std::vector<std::shared_ptr<Job>> &jobs);
  std::unique_lock<std::mutex> Lock() {
    if (Enable()) {
      return std::unique_lock{mut_};
    } else {
      return std::unique_lock<std::mutex>{};
    }
  }

  friend class InferModelStore;

 private:
  struct CacheItem {
    Model *model;
    bool cached;
    std::mutex mut;
  };

  static void Init();
  void MaybeAddCacheItem(const std::string &name, Model *model);
  void ReserveCacheInternal(const std::string &name, size_t rank,
                            std::unique_lock<std::mutex> &reserved_lock);

  static std::vector<std::unique_ptr<WarmModelCache>> warm_model_caches_;

  std::mutex mut_;
  std::condition_variable fifo_cv_;
  size_t cached_nbytes_;
  int device_id_;
  std::unordered_map<std::string, std::unique_ptr<CacheItem>> warm_cache_items_;

};

class ColdModelCache {
 private:
  static std::vector<std::unique_ptr<ColdModelCache>> cold_model_caches_;

  struct CacheItem {
    Model *model;
    std::vector<size_t> cached_groups_id;
    size_t cached_group_nbytes;
  };

  int device_id_;
  size_t current_cached_nbytes_;
  size_t current_capacity_nbytes_;
  std::mutex mut_;
  std::unordered_map<
      std::string, std::unique_ptr<CacheItem>
  > cold_cache_items_;

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

  ColdModelCache(int device_id)
    : current_cached_nbytes_{0}, 
      current_capacity_nbytes_(Config::cold_cache_min_capacity_nbytes), 
      device_id_{device_id} {}

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
  std::tuple<
      std::vector<size_t>, 
      std::vector<std::pair<std::string, std::vector<size_t>>>, 
      bool>
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
  PopCacheItem(const std::string& name, size_t rank, 
               bool pop_to_inference, 
               std::unique_lock<std::mutex> &lock);

  /**
   * Retrieves the list of models that are eligible for eviction from the infer model store.
   * 
   * @param capacity The desired capacity after eviction.
   * @param ignore_models 
   * @param lock A unique lock on the mutex.
   * @return The list of models that should be evicted.
   */
  evict_list GetEvictModels(memory_byte_t capacity, 
                            std::array<Model*, 2> ignore_models, 
                            std::unique_lock<std::mutex>& lock);

  std::unique_lock<std::mutex> Lock() {
    return std::unique_lock{mut_};
  }

  std::pair<size_t, size_t> capacity_and_cache = {0, 0};
  std::pair<size_t, size_t> GetColdCacheCapacityAndCache() {
    return {current_capacity_nbytes_, current_cached_nbytes_};
    // if (capacity_and_cache == std::pair<size_t, size_t>{0, 0}) {
    //   return {current_capacity_nbytes_, current_cached_nbytes_};
    // } else {
    //   return capacity_and_cache;
    // }
  }

  void BlockProfilter() {
    capacity_and_cache = {current_capacity_nbytes_, current_cached_nbytes_};
  }

  void UnblockProfilter() {
    capacity_and_cache = {0, 0};
  }

  inline size_t GetCachedNbytes(std::unique_lock<std::mutex> &lock) {
    return current_cached_nbytes_;
  }

  bool TakeSpace(memory_byte_t nbytes);

  inline size_t GetCachedNbytesUnsafe() {
    return current_cached_nbytes_;
  }

  inline size_t GetColdCacheReleasableMemoryMBUnsafe() {
    if (current_cached_nbytes_ > Config::cold_cache_min_capacity_nbytes) {
      return sta::ByteToMB(current_cached_nbytes_ 
                           - Config::cold_cache_min_capacity_nbytes);
    } else {
      return 0;
    }
  }

  double GetBufferMBUnsafe();

  double GetCacheSizeMBUnsafe();

  void SetNewCapacity(memory_byte_t new_capacity,
                      std::unique_lock<std::mutex> &lock);

  inline memory_byte_t GetCacheCapacity(std::unique_lock<std::mutex> &lock) {
    return current_capacity_nbytes_;
  }

  inline double GetFreeMemoryWithCacheEmpty(
      double free_memory_MB, 
      std::unique_lock<std::mutex> &lock) {
    if (current_cached_nbytes_ > Config::cold_cache_min_capacity_nbytes) {
      free_memory_MB += sta::ByteToMB(current_cached_nbytes_ 
                                      - Config::cold_cache_min_capacity_nbytes);
    }
    // LOG(INFO) << "[ColdModelCache] FreeMemory " << free_memory_MB << "MB";
    return free_memory_MB;
  }

  inline double GetFreeMemoryWithCacheReserve(
        double free_memory_MB, 
        std::unique_lock<std::mutex> &lock) {
    free_memory_MB -= sta::ByteToMB(current_capacity_nbytes_);
    if (free_memory_MB < 0) {
      free_memory_MB = 0;
    }
    // LOG(INFO) << "[ColdModelCache] FreeMemory " << free_memory_MB << "MB";
    return free_memory_MB;
  }

  double GetReleaseReserveMemoryMBUnsafe();
  double GetAdjustReserveMemoryMBUnsafe();
  double GetReleaseReserveMemoryMB(std::unique_lock<std::mutex> &lock);  
  double GetAdjustReserveMemoryMB(std::unique_lock<std::mutex> &lock);  

  inline static memory_byte_t GetStablePointCapacity() {
    return (Config::cold_cache_min_capacity_nbytes + 
           Config::cold_cache_max_capacity_nbytes) / 2;
  }

  std::string PrintCacheInfo(std::unique_lock<std::mutex> &lock); 

  static void Init();
  static ColdModelCache * Get(int device_id);
};

}

#endif