#include "workload.h"
#include "triton_impl.h"
#include "workload.cc.in"
#include "workload.h.in"
#include <memory>

namespace colserve::workload {

std::unique_ptr<IWorkload> GetTritonWorkload(std::shared_ptr<grpc::Channel> channel,
       std::chrono::seconds duration, double delay_before_profile,
       const std::string &infer_timeline) {
  return std::make_unique<colserve::workload::COLSYS_CLIENT_IMPL_NAMESPACE::Workload>(channel, duration, delay_before_profile, infer_timeline);
}
grpc::Status triton_backend::ServeStub::Inference(grpc::ClientContext *context,
                                                  const InferRequest &request,
                                                  InferResult *response) {
  inference::ModelInferRequest req;
  WarmCache::IncModel(*stub_, context, request.model());
  stub_->ModelInfer(context, request.value, &response->value);
  WarmCache::DecModel(*stub_, context, request.model());
  return grpc::Status::OK;
}
} // namespace colserve::workload