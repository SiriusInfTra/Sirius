#include "warm_cache.h"
#include <algorithm>
#include <mutex>


namespace colserve::workload {
  std::mutex WarmCache::g_mutex;
  std::mutex WarmCache::m_mutex;
  std::unordered_map<std::string, std::unique_ptr<WarmCache>> WarmCache::loaded_models;

  void WarmCache::IncModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *_, const std::string &model_name) {
    LOG(INFO) << "[TritonProxy] Model " << model_name << " inc.";

    std::unique_lock g_lock{g_mutex};
    LOG(INFO) << "[TritonProxy] Model " << model_name << " inc: lock g_mutex #1.";
    WarmCache *warm_cache;
    {
      std::unique_lock m_lock{m_mutex};
      auto iter = loaded_models.find(model_name);
      if (iter == loaded_models.cend()) {
        warm_cache = new WarmCache(model_name);
        loaded_models.insert({model_name, std::unique_ptr<WarmCache>(warm_cache)});
      } else {
        warm_cache = iter->second.get(); 
      }
    }
    std::unique_lock s_lock{warm_cache->s_mutex_};
    warm_cache->infering_cnt_++;
    warm_cache->hotness_++;
    if (warm_cache->alive_) {
      LOG(INFO) << "[TritonProxy] Model " << model_name << " alive.";
      return;
    }
    std::vector<WarmCache*> alive_models;
    for (auto &&[_, warm_cache] : loaded_models) {
      if (warm_cache->alive_) {
        CHECK_NE(warm_cache->model_name_, model_name);
        alive_models.push_back(warm_cache.get());
      }
    }
    warm_cache->alive_ = true; // must set alive before unlock g_mutex
    LOG(INFO) << "[TritonProxy] Model " << model_name << "alive models: " << alive_models.size();
    if (alive_models.size() >= 8) {
      LOG(INFO) << "[TritonProxy] Model " << model_name
                << " dead, no rooms.";
      std::sort(alive_models.begin(), alive_models.end(), [](auto &a, auto &b) {
        return a->hotness_ < b->hotness_;
      });
      CHECK_GT(alive_models.size(), 0);
      auto *evict_model = alive_models[0];
      std::unique_lock e_lock{evict_model->s_mutex_};
      g_lock.unlock();
      LOG(INFO) << "[TritonProxy] Model " << model_name << " unlock g_mutex #1.";
      LOG(INFO) << "[TritonProxy] Model " << model_name << ": wait for " << evict_model->model_name_;
      evict_model->free_cond_.wait(e_lock, [&evict_model] {
        return evict_model->infering_cnt_ == 0;
      });
      LOG(INFO) << "[TritonProxy] Model " << model_name << ": wait done " << evict_model->model_name_;
      ::inference::RepositoryModelUnloadRequest request;
      ::inference::RepositoryModelUnloadResponse response;
      request.set_model_name(evict_model->model_name_);
      ::grpc::ClientContext context;
      auto status = stub.RepositoryModelUnload(&context, request, &response);
      CHECK(status.ok());
      evict_model->alive_ = false;
      LOG(INFO) << "[TritonProxy] Model " << model_name << ": evit " << evict_model->model_name_ << " unloaded.";
    } else {
      g_lock.unlock();
      LOG(INFO) << "[TritonProxy] Model " << model_name << " unlock g_mutex #2.";
    }
    ::inference::RepositoryModelLoadRequest request;
    ::inference::RepositoryModelLoadResponse response;
    ::grpc::ClientContext context;
    request.set_model_name(model_name);
    auto status = stub.RepositoryModelLoad(&context, request, &response);
    CHECK(status.ok());

    LOG(INFO) << "[TritonProxy] Model " << model_name << " loaded.";
  }

  void WarmCache::DecModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *context, const std::string &model_name) {
    LOG(INFO) << "[TritonProxy] Model " << model_name << " dec.";
    std::unique_lock g_lock{m_mutex};
    LOG(INFO) << "[TritonProxy] Model " << model_name << " dec: lock g_mutex #2.";
    auto iter = loaded_models.find(model_name);
    auto warm_cache = iter->second.get();
    std::unique_lock s_lock{warm_cache->s_mutex_};
    CHECK(iter != loaded_models.cend());
    CHECK_GT(warm_cache->infering_cnt_, 0);
    warm_cache->infering_cnt_ -= 1;
    LOG(INFO) << "[TritonProxy] Model " << model_name << " dec: " << warm_cache->infering_cnt_;
    if (warm_cache->infering_cnt_ == 0) {
      warm_cache->free_cond_.notify_all();
    } 
  }

} // colserve::workload