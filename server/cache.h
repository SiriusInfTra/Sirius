#ifndef COLSERVE_CACHE_H
#define COLSERVE_CACHE_H
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unordered_map>
#include <tvm/graph_executor.h>
#include <sstream>
#include <mutex>
#include "profiler.h"
#include "glog/logging.h"

namespace colserve {

const constexpr char CACHE_LOG_PREFIX[] = "[Cache]";

struct CacheItem {
  tvm::GraphExecutor *graph_executor;
  bool is_cache;
  std::string name;
};


class LRUPolicy {
private:
  std::mutex mutex_;
  std::list<CacheItem> lru_list_;
  std::unordered_map<tvm::GraphExecutor*, std::list<CacheItem>::iterator> cache_;
  std::unordered_map<tvm::GraphExecutor*, std::mutex> loading_locks_;
  size_t cached_nbytes_ = 0;


  inline void DumpLRUList(const std::string &tag) const {
    std::stringstream ss;
    ss << "[";
    for(auto && cache_item : lru_list_) {
      ss << cache_item.name << ", ";
    }
    ss << "]";
    LOG(INFO) << CACHE_LOG_PREFIX << " LRU List(" << tag <<"): " << ss.str() << std::endl;
  }

public:

  CacheItem RemoveIfPresent(const std::string &name, tvm::GraphExecutor *graph_executor) {
    std::unique_lock lock{mutex_};
    DumpLRUList("RemoveIfPresent-" + name + "-begin");
    auto iter = cache_.find(graph_executor); 
    if (iter == cache_.end()) {
      DumpLRUList("RemoveIfPresent-" + name + "-end1");
      std::unique_lock loading_locker{loading_locks_[graph_executor]};
      lock.unlock();
      graph_executor->Init();
      return {CacheItem{graph_executor, false, name}};
    }
    auto list_node = iter->second;
    auto cache_item = *list_node;
    lru_list_.erase(list_node);
    cache_.erase(iter);
    CHECK_EQ(cache_item.is_cache, true);
    cached_nbytes_ -= cache_item.graph_executor->GetStorageSize();
    DumpLRUList("RemoveIfPresent-" + name + "-end2");

    return cache_item;
  }

  std::tuple<CacheItem, size_t> PutRemoveLast(const std::string &name, tvm::GraphExecutor *graph_executor, size_t max_cache_nbytes) {
    std::unique_lock lock{mutex_};
    DumpLRUList("PutRemoveLast-" + name + "-begin");
    CHECK_LE(graph_executor->GetStorageSize(), max_cache_nbytes);
    if (cached_nbytes_ + graph_executor->GetStorageSize() > max_cache_nbytes) { /* not capable to hold  */
      CHECK_NE(lru_list_.empty(), true);
      auto cache_item = lru_list_.back();
      lru_list_.pop_back();
      cached_nbytes_ -= cache_item.graph_executor->GetStorageSize();
      CHECK_EQ(cache_.erase(cache_item.graph_executor), 1);
      DumpLRUList("PutRemoveLast-" + name + "-end1");
      std::unique_lock loading_locker{loading_locks_[cache_item.graph_executor]};
      lock.unlock();
      cache_item.graph_executor->DeInit();
      return {cache_item, cached_nbytes_};
    }
    auto cache_item = CacheItem{graph_executor, true, name};
    lru_list_.push_front(cache_item);
    cache_.emplace(std::make_pair(graph_executor, lru_list_.begin()));
    cached_nbytes_ += graph_executor->GetStorageSize();
    DumpLRUList("PutRemoveLast-" + name + "-end2");
    return {cache_item, cached_nbytes_};
  }

};

class GraphCache {
private:
  LRUPolicy policy_;
  static std::unique_ptr<GraphCache> instance_;
  size_t max_cache_nbytes_;
public:
  static void Init(size_t nbytes);
  static GraphCache* Get();

  GraphCache(size_t nbytes): max_cache_nbytes_(nbytes) { }

  ~GraphCache() {
    ReleaseAll();
  }
  
  // assume graph_executor is accessed by a unique thread
  void InitGraphExecutor(const std::string &name, tvm::GraphExecutor* graph_executor) {
    auto cache_item = policy_.RemoveIfPresent(name, graph_executor);
    if (cache_item.is_cache) { /* cached, already inited*/
      LOG(INFO) << CACHE_LOG_PREFIX << " load model: " << name << ", found it already in cache, skip init.";
      Profiler::Get()->RecordPerf(Profiler::PerfItem::InferModelLoad, 1.0);
    } else {
      LOG(INFO) << CACHE_LOG_PREFIX << " load model: " << name << ", found it not in cache, init it.";
      Profiler::Get()->RecordPerf(Profiler::PerfItem::InferModelLoad, 0.0);
    }
  }

  void DeInitGraphExecutor(const std::string &name, tvm::GraphExecutor* graph_executor) {
    std::stringstream log_prefix;
    log_prefix << CACHE_LOG_PREFIX << " model " << name << " with graph size " << graph_executor->GetStorageSize() << " deinit.";
    if (graph_executor->GetStorageSize() > max_cache_nbytes_) {
      LOG(INFO) << log_prefix.str() << " cache nbytes: " << max_cache_nbytes_ << ". cannot cache.";
      graph_executor->DeInit();
      return;
    }
    while (true) {
      auto [cache_item, cached_nbytes] = policy_.PutRemoveLast(name, graph_executor, max_cache_nbytes_);
      if (cache_item.graph_executor == graph_executor) {
        LOG(INFO) << log_prefix.str() << " put it into cache, current cache nbyte=" << cached_nbytes << ".";
        break;
      }
      LOG(INFO) << log_prefix.str() << " no space to hold it. so remove " << cache_item.name << "(" << cache_item.graph_executor->GetStorageSize() << "), current cache nbyte=" << cached_nbytes << ".";
    }
  }

  void ReleaseAll() {
    LOG(INFO) << CACHE_LOG_PREFIX << " release all.";
    // while(policy_.GetStoragesNBytes() > 0) {
    //   auto cache_item = policy_.Remove();
    //   cache_item.graph_executor->DeInit();
    // }
  }
};

}


#endif