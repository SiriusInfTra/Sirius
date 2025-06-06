#include "warm_cache.h"
#include <chrono>
#include <fstream>
#include <atomic>
#include <glog/logging.h>
#include <mutex>

namespace colserve::workload {

std::mutex WarmCache::swapping_mutex_;
std::mutex WarmCache::data_mutex_;
std::unordered_map<std::string, std::unique_ptr<WarmCache>> WarmCache::loaded_models_;
std::atomic<int> WarmCache::concurrent_loads[MAX_DEVICE];
TritonConfig WarmCache::triton_config_;
size_t WarmCache::curr_memory_usage_[MAX_DEVICE];

void WarmCache::Init(TritonConfig config) {
  triton_config_ = std::move(config);
  if (triton_config_.max_memory_nbytes == 0) {
    LOG(INFO) << "[TritonProxy] No memory limit.";
  } else {
    LOG(INFO) << "[TritonProxy] Memory limit: "
              << triton_config_.max_memory_nbytes << " MB.";
  }
  for (auto&& curr_memory_usage : curr_memory_usage_) {
    curr_memory_usage = 0;
  }
}

void WarmCache::IncModel(inference::GRPCInferenceService::Stub& stub,
                         ::grpc::ClientContext* _,
                         const std::string& model_name) {
  if (triton_config_.max_memory_nbytes == 0) {
    return;
  }
  LOG(INFO) << "[TritonProxy] Model " << model_name << " inc.";

  //-------------
  // 1. GET_ITEM
  //-------------
  auto t0 = std::chrono::steady_clock::now();
  WarmCache* warm_cache;
  {
    std::unique_lock m_lock{data_mutex_};
    auto iter = loaded_models_.find(model_name);
    if (iter == loaded_models_.cend()) {
      warm_cache = new WarmCache(model_name);
      loaded_models_.insert({model_name, std::unique_ptr<WarmCache>(warm_cache)});
    } else {
      warm_cache = iter->second.get();
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  LOG(INFO) << "[TritonProxy] MODEL " << model_name << " GET_ITEM "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
            << "ms.";

  //-------------
  // 2. GET_LOCK
  //-------------
  size_t model_memory_usage = GetModelMemoryUsage(model_name);
  {
    // wait if someone is wait model to release
    // std::unique_lock inc_lock{warm_cache->inc_mutex_};
  }
  std::unique_lock s_lock{warm_cache->s_mutex_};
  size_t& curr_memory_usage_device = curr_memory_usage_[GetModelDevice(model_name)];
  auto t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "[TritonProxy] MODEL " << model_name << " GET_LOCK "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms.";

  //-------------
  // 3. SWAPPING
  //-------------
  if (!warm_cache->alive_) {
    LOG(INFO) << "[TritonProxy] Model " << model_name << " not alive.";
    s_lock.unlock();
    std::unique_lock swapping_lock{swapping_mutex_};
    s_lock.lock();
    auto t3 = std::chrono::steady_clock::now();
    LOG(INFO) << "[TritonProxy] MODEL " << model_name << " GET_SWAP_LOCK "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << "ms.";
    // -----------------------
    // 3.1 EVICT Other Models
    // -----------------------
    while (curr_memory_usage_device + model_memory_usage > triton_config_.max_memory_nbytes) {
      WarmCache* evict_model = nullptr;
      std::unique_lock<std::mutex> evict_lock;
      for (bool try_lock : {true, false}) {
        for (auto&& [_, e_model] : loaded_models_) {
          if (e_model->model_name_ == model_name) {
            continue;
          }
          std::unique_lock<std::mutex> e_lock;
          if (try_lock) {
            e_lock = std::unique_lock<std::mutex>(e_model->s_mutex_, std::try_to_lock);
            if (!e_lock.owns_lock()) {
              continue;
            }
          } else {
            e_lock = std::unique_lock<std::mutex>(e_model->s_mutex_);
          }
          if (e_model->alive_ && (evict_model == nullptr ||
              evict_model->hotness_ > e_model->hotness_)) {
            evict_model = e_model.get();
            evict_lock = std::move(e_lock);
          }
        }
        if (evict_model != nullptr) {
          // try lock success
          break;
        }
      }

      DLOG(INFO) << "[TritonProxy] Model " << model_name
                << " try to evict model " << evict_model->model_name_ << ".";
      auto t4 = std::chrono::steady_clock::now();
      DLOG(INFO) << "[TritonProxy] MODEL " << model_name << " FIND_EVICT_MODEL "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
                << "ms.";

      // Wait for the evicted model to finish inference
      evict_model->free_cond_.wait(evict_lock, [&]() {
          return evict_model->infering_cnt_ == 0;
      });
      auto t5 = std::chrono::steady_clock::now();
      DLOG(INFO) << "[TritonProxy] Model " << model_name << " WAIT_MODEL_RELEASE "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
                << "ms.";

      // Unload the evicted model
      ::inference::RepositoryModelUnloadRequest request;
      ::inference::RepositoryModelUnloadResponse response;
      request.set_model_name(evict_model->model_name_);
      ::grpc::ClientContext context;
      auto status = stub.RepositoryModelUnload(&context, request, &response);
      CHECK(status.ok());
      evict_model->alive_ = false;
      curr_memory_usage_device -= GetModelMemoryUsage(evict_model->model_name_);
      auto t6 = std::chrono::steady_clock::now();
      DLOG(INFO) << "[TritonProxy] MODEL " << model_name << " UNLOAD_MODEL "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count()
                << "ms.";
    }
    //---------------
    // 3.2 LOAD Self
    //---------------
    auto t7 = std::chrono::steady_clock::now();
    warm_cache->alive_ = true;
    curr_memory_usage_device += model_memory_usage;

    auto& concurrent_load =
        concurrent_loads[triton_config_.models_device.find(model_name)->second];
    if (concurrent_load.fetch_add(1) < MAX_CONCURRENT_LOAD) {
      swapping_lock.unlock();
    }

    ::inference::RepositoryModelLoadRequest request;
    ::inference::RepositoryModelLoadResponse response;
    ::grpc::ClientContext context;
    request.set_model_name(model_name);
    auto status = stub.RepositoryModelLoad(&context, request, &response);
    CHECK(status.ok());
    concurrent_load.fetch_sub(1);
    auto t8 = std::chrono::steady_clock::now();
    LOG(INFO) << "[TritonProxy] MODEL " << model_name << " LOAD_MODEL "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() 
              << "ms. " << " TOT_INC "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t0).count() 
              << "ms.";
    LOG(INFO) << "[TritonProxy] Model " << model_name
              << " loaded, predict curr_memory_usage_: " << curr_memory_usage_device;
  }

  warm_cache->infering_cnt_++;
  warm_cache->hotness_++;
  LOG(INFO) << "[TritonProxy] Model " << model_name << " inc done.";
}

void WarmCache::DecModel(inference::GRPCInferenceService::Stub& stub,
                         ::grpc::ClientContext* context,
                         const std::string& model_name) {
  if (triton_config_.max_memory_nbytes == 0) {
    return;
  }
  LOG(INFO) << "[TritonProxy] Model " << model_name << " dec.";

  std::unique_lock m_lock{data_mutex_};
  auto iter = loaded_models_.find(model_name);
  CHECK(iter != loaded_models_.cend());
  auto warm_cache = iter->second.get();

  std::unique_lock s_lock{warm_cache->s_mutex_};
  CHECK_GT(warm_cache->infering_cnt_, 0);
  warm_cache->infering_cnt_ -= 1;
  LOG(INFO) << "[TritonProxy] Model " << model_name
            << " dec: " << warm_cache->infering_cnt_;

  if (warm_cache->infering_cnt_ == 0) {
    warm_cache->free_cond_.notify_all();
  }
}

size_t WarmCache::GetModelMemoryUsage(const std::string& name) {
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

size_t WarmCache::GetModelDevice(const std::string& name) {
  if (triton_config_.max_memory_nbytes == 0) {
    return 0;
  }
  auto it = triton_config_.models_device.find(name);
  CHECK(it != triton_config_.models_device.end())
      << "Model " << name << " not found in config";
  return it->second;
}

TritonConfig TritonConfig::LoadConfig(const std::string& filepath,
                                      size_t max_memory_nbytes,
                                      const std::string& model_device_config_path) {
  std::ifstream file(filepath);
  CHECK(file.is_open()) << "Unable open configuration: " << filepath;

  TritonConfig config;
  config.max_memory_nbytes = max_memory_nbytes;

  std::string line;
  while (std::getline(file, line)) {
    // Remove spaces and comments
    line = line.substr(0, line.find('#'));
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);

    if (line.empty())
      continue;

    auto delimiter_pos = line.find('=');
    CHECK(delimiter_pos != std::string::npos) << "wrong configuration: " << line;

    std::string key = line.substr(0, delimiter_pos);
    std::string value_str = line.substr(delimiter_pos + 1);

    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value_str.erase(0, value_str.find_first_not_of(" \t"));
    value_str.erase(value_str.find_last_not_of(" \t") + 1);

    size_t value;
    try {
      value = std::stoull(value_str);
    } catch (const std::invalid_argument& e) {
      LOG(FATAL) << "invalid number " << value_str;
    } catch (const std::out_of_range& e) {
      LOG(FATAL) << "out of range " << value_str;
    }

    config.models_memory_nbytes[key] = value;
  }

  file.close();

  std::ifstream device_map_file(model_device_config_path);
  CHECK(device_map_file.is_open())
      << "Unable open device map configuration: " << model_device_config_path;
  std::string model_name;
  int model_device;
  while (device_map_file >> model_name >> model_device) {
    config.models_device[model_name] = model_device;
  }
  return config;
}

}  // namespace colserve::workload