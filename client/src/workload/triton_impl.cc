#include "workload.h"
#include "triton_impl.h"
#include "warm_cache.h"
#include "workload.cc.in"
#include "workload.h.in"
#include <memory>

namespace colserve::workload {

std::unique_ptr<IWorkload> GetTritonWorkload(
    std::shared_ptr<grpc::Channel> channel,
    std::chrono::seconds duration, double delay_before_profile,
    const std::string &infer_timeline, size_t max_memory_nbytes, 
    const std::string &triton_config, const std::string &triton_device_map) {
  WarmCache::Init(TritonConfig::LoadConfig(
      triton_config, max_memory_nbytes, triton_device_map));
  return std::make_unique<colserve::workload::SIRIUS_CLIENT_IMPL_NAMESPACE::Workload>(
      channel, duration, delay_before_profile, infer_timeline);
}

grpc::Status triton_backend::ServeStub::Inference(
    grpc::ClientContext *context,
    const InferRequest &request,
    InferResult *response) {
  inference::ModelInferRequest req;
  WarmCache::IncModel(*stub_, context, request.model());
  stub_->ModelInfer(context, request.value, &response->value);
  WarmCache::DecModel(*stub_, context, request.model());
  return grpc::Status::OK;
}

} // namespace colserve::workload