#ifndef COLSYS_WARMUP_H__
#define COLSYS_WARMUP_H__
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <glog/logging.h>
#include <grpc_service.grpc.pb.h>
namespace colserve::workload {
class WarmCache {
 private:
  static std::mutex g_mutex;
  static std::mutex m_mutex;
  static std::unordered_map<std::string, std::unique_ptr<WarmCache>> loaded_models;

  std::mutex s_mutex_;
  std::condition_variable free_cond_;
  std::string model_name_;
  size_t hotness_;
  bool alive_;
  int infering_cnt_;

 public:
  WarmCache(const std::string &model_name) : model_name_(model_name), hotness_(0), infering_cnt_(0), alive_(false) {}
  static void IncModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *context, const std::string &model_name);
  static void DecModel(inference::GRPCInferenceService::Stub &stub, ::grpc::ClientContext *context, const std::string &model_name);
};
}
#endif // 