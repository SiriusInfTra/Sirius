#include "warm_cache.h"
#include <algorithm>
#include <mutex>


namespace colserve::workload {
  std::mutex WarmCache::g_mutex_;
  std::mutex WarmCache::m_mutex_;
  std::unordered_map<std::string, std::unique_ptr<WarmCache>> WarmCache::loaded_models_;
  TritonConfig WarmCache::triton_config_;
  size_t WarmCache::curr_memory_usage_ = 0;
  
  void WarmCache::SetTritonConfig(TritonConfig config) {
    triton_config_ = std::move(config);
  }

  void WarmCache::IncModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *_, const std::string &model_name) {
    LOG(INFO) << "[TritonProxy] Model " << model_name << " inc.";

    std::unique_lock g_lock{g_mutex_};
    LOG(INFO) << "[TritonProxy] Model " << model_name << " inc: lock g_mutex #1.";
    WarmCache *warm_cache;
    {
      std::unique_lock m_lock{m_mutex_};
      auto iter = loaded_models_.find(model_name);
      if (iter == loaded_models_.cend()) {
        warm_cache = new WarmCache(model_name);
        loaded_models_.insert({model_name, std::unique_ptr<WarmCache>(warm_cache)});
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
    for (auto &&[_, warm_cache] : loaded_models_) {
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
      g_lock.unlock();
      auto *evict_model = alive_models[0];
      std::unique_lock e_lock{evict_model->s_mutex_};

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
    std::unique_lock g_lock{m_mutex_};
    LOG(INFO) << "[TritonProxy] Model " << model_name << " dec: lock g_mutex #2.";
    auto iter = loaded_models_.find(model_name);
    auto warm_cache = iter->second.get();
    std::unique_lock s_lock{warm_cache->s_mutex_};
    CHECK(iter != loaded_models_.cend());
    CHECK_GT(warm_cache->infering_cnt_, 0);
    warm_cache->infering_cnt_ -= 1;
    LOG(INFO) << "[TritonProxy] Model " << model_name << " dec: " << warm_cache->infering_cnt_;
    if (warm_cache->infering_cnt_ == 0) {
      warm_cache->free_cond_.notify_all();
    } 
  }

  size_t WarmCache::GetModelMemoryUsage(const std::string &name) {
    std::string name_normalized{name.cbegin(), name.cbegin() + name.find('-')};
    auto it = triton_config_.models_memory_nbytes.find(name_normalized);
    CHECK(it != triton_config_.models_memory_nbytes.end())
        << "Model " << name << " not found in config";
    return it->second;
  }
  } // namespace colserve::workload