#include "workload.h"
#include "colsys_grpc.h"
#include "workload.cc.in"
#include "workload.h.in"
#include <memory>

namespace colserve::workload {

std::unique_ptr<IWorkload> GetColsysWorkload(std::shared_ptr<grpc::Channel> channel,
       std::chrono::seconds duration, double delay_before_profile,
       const std::string &infer_timeline) {
  return std::make_unique<colserve::workload::COLSYS_CLIENT_IMPL_NAMESPACE::Workload>(channel, duration, delay_before_profile, infer_timeline);
}
}