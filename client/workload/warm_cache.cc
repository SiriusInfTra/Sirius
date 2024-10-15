#include "warm_cache.h"
#include <fstream>
#include <atomic>
#include <glog/logging.h>
#include <mutex>


namespace colserve::workload {
std::mutex WarmCache::swapping_mutex_;
std::mutex WarmCache::m_mutex_;
std::unordered_map<std::string, std::unique_ptr<WarmCache>> WarmCache::loaded_models_;
TritonConfig WarmCache::triton_config_;
size_t WarmCache::curr_memory_usage_ = 0;

void WarmCache::SetTritonConfig(TritonConfig config) {
  triton_config_ = std::move(config);
  if (triton_config_.max_memory_nbytes == 0) {
    LOG(INFO) << "[TritonProxy] No memory limit.";
  } else {
    LOG(INFO) << "[TritonProxy] Memory limit: " << triton_config_.max_memory_nbytes << " MB. ";
  }
}

void WarmCache::IncModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *_, const std::string &model_name) {
  LOG(INFO) << "[TritonProxy] Model " << model_name << " inc.";
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
  size_t model_memory_usage = GetModelMemoryUsage(model_name);
  
  std::unique_lock s_lock{warm_cache->s_mutex_};

  std::unique_lock swapping_lock{swapping_mutex_, std::defer_lock};
  if (!warm_cache->alive_) {
    LOG(INFO) << "[TritonProxy] Model " << model_name << " not alive.";
    s_lock.unlock();
    swapping_lock.lock();
    s_lock.lock();
    while(curr_memory_usage_ + model_memory_usage > triton_config_.max_memory_nbytes) {
      WarmCache *evict_model = nullptr;
      std::unique_lock<std::mutex> evict_lock;
      for (auto &&[_, model] : loaded_models_) {
        if (model->model_name_ == model_name) {
          continue;
        }
        std::unique_lock s_lock{model->s_mutex_};
        if (model->alive_ && (evict_model == nullptr 
          || evict_model->hotness_ > model->hotness_)) {
          evict_model = model.get();
          evict_lock = std::move(s_lock);
        }
      }
      LOG(INFO) << "[TritonProxy] Model " << model_name <<  " try to evict model " << evict_model->model_name_ << ".";
      evict_model->free_cond_.wait(evict_lock, [&]() { return evict_model->infering_cnt_ == 0; });
      LOG(INFO) << "[TritonProxy] Model " << model_name <<  " evict model " << evict_model->model_name_ << ": wait done.";
      ::inference::RepositoryModelUnloadRequest request;
      ::inference::RepositoryModelUnloadResponse response;
      request.set_model_name(evict_model->model_name_);
      ::grpc::ClientContext context;
      auto status = stub.RepositoryModelUnload(&context, request, &response);
      CHECK(status.ok());
      evict_model->alive_ = false;
      curr_memory_usage_ -= GetModelMemoryUsage(evict_model->model_name_);
    }
    ::inference::RepositoryModelLoadRequest request;
    ::inference::RepositoryModelLoadResponse response;
    ::grpc::ClientContext context;
    request.set_model_name(model_name);
    auto status = stub.RepositoryModelLoad(&context, request, &response);
    CHECK(status.ok());
    warm_cache->alive_ = true;
    curr_memory_usage_ += model_memory_usage;
    LOG(INFO) << "[TritonProxy] Model " << model_name << " loaded, predict curr_memory_usage_: " << curr_memory_usage_;
  }
  warm_cache->infering_cnt_++;
  warm_cache->hotness_++;
  LOG(INFO) << "[TritonProxy] Model " << model_name << " inc done.";
}

void WarmCache::DecModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *context, const std::string &model_name) {
  LOG(INFO) << "[TritonProxy] Model " << model_name << " dec.";
  std::unique_lock m_lock{m_mutex_};
  auto iter = loaded_models_.find(model_name);
  CHECK(iter != loaded_models_.cend());
  auto warm_cache = iter->second.get();
  std::unique_lock s_lock{warm_cache->s_mutex_};
  CHECK_GT(warm_cache->infering_cnt_, 0);
  warm_cache->infering_cnt_ -= 1;
  LOG(INFO) << "[TritonProxy] Model " << model_name << " dec: " << warm_cache->infering_cnt_;
  if (warm_cache->infering_cnt_ == 0) {
    warm_cache->free_cond_.notify_all();
  } 
}

size_t WarmCache::GetModelMemoryUsage(const std::string &name) {
  if (triton_config_.max_memory_nbytes == 0) {
    return 0;
  }
  size_t pos = name.find('-');
  std::string name_normalized = name.substr(0, pos);
  auto it = triton_config_.models_memory_nbytes.find(name_normalized);
  CHECK(it != triton_config_.models_memory_nbytes.end())
      << "Model " << name << " not found in config";
  return it->second;
}

TritonConfig TritonConfig::LoadConfig(const std::string &filepath,
                                      size_t max_memory_nbytes) {
  std::ifstream file(filepath);
  CHECK(file.is_open()) << "无法打开配置文件: " << filepath;

  TritonConfig config;
  config.max_memory_nbytes = max_memory_nbytes;

  std::string line;
  while (std::getline(file, line)) {
    // 去除空格和注释
    line = line.substr(0, line.find('#'));
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);

    if (line.empty())
      continue;

    auto delimiter_pos = line.find('=');
    CHECK(delimiter_pos != std::string::npos) << "配置格式错误: " << line;

    std::string key = line.substr(0, delimiter_pos);
    std::string value_str = line.substr(delimiter_pos + 1);

    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value_str.erase(0, value_str.find_first_not_of(" \t"));
    value_str.erase(value_str.find_last_not_of(" \t") + 1);

    size_t value;
    try {
      value = std::stoull(value_str);
    } catch (const std::invalid_argument &e) {
      LOG(FATAL) << "无效的数字: " << value_str;
    } catch (const std::out_of_range &e) {
      LOG(FATAL) << "数字超出范围: " << value_str;
    }

    config.models_memory_nbytes[key] = value;
  }

  file.close();
  return config;
}
} // namespace colserve::workload