#ifndef SIRIUS_WARMUP_H__
#define SIRIUS_WARMUP_H__

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <glog/logging.h>
#include <grpc_service.grpc.pb.h>


namespace colserve::workload {

struct TritonConfig {
  size_t max_memory_nbytes;
  std::unordered_map<std::string, size_t> models_memory_nbytes;
  std::unordered_map<std::string, int> models_device;

  static TritonConfig LoadConfig(
      const std::string &model_config_path,
      size_t max_memory_nbytes, 
      const std::string &model_device_config_path);
};

class WarmCache {
  const constexpr static size_t MAX_DEVICE = 4;
  const constexpr static int MAX_CONCURRENT_LOAD = 2;

 private:
  static std::mutex swapping_mutex_; // ordered request
  static std::mutex data_mutex_; // protect datastruct access
  static std::unordered_map<std::string, std::unique_ptr<WarmCache>> loaded_models_;
  static std::atomic<int> concurrent_loads[MAX_DEVICE];
  static TritonConfig triton_config_;
  static size_t curr_memory_usage_[MAX_DEVICE];

  std::mutex s_mutex_;
  std::mutex inc_mutex_;
  std::condition_variable free_cond_;
  std::string model_name_;
  size_t hotness_;
  bool alive_;
  int infering_cnt_;

  static size_t GetModelMemoryUsage(const std::string &name);
  static size_t GetModelDevice(const std::string &name);

public:
  WarmCache(const std::string &model_name) 
      : model_name_(model_name), hotness_(0), infering_cnt_(0), alive_(false) {}

  static void Init(TritonConfig config);
  static void IncModel(inference::GRPCInferenceService::Stub &stub, 
                       ::grpc::ClientContext *context, 
                       const std::string &model_name);
  static void DecModel(inference::GRPCInferenceService::Stub &stub, 
                       ::grpc::ClientContext *context, 
                       const std::string &model_name);
};
} // namespace colserve::workload

#endif // 