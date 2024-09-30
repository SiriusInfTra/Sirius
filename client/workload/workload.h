#ifndef COLSYS_IWORKLOAD_H__
#define COLSYS_IWORKLOAD_H__

#include <grpcpp/channel.h>
#include <memory>
#include <string>
#include <functional>
#include <chrono>

namespace colserve::workload {

using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
using double_ms_t = std::chrono::duration<double, std::milli>;

class IWorkload {
public:
  virtual bool Hello() = 0;
  virtual bool InferenceWorkloadStart() = 0;
  virtual void InferenceWorkloadDone() = 0;
  virtual void Run() = 0;
  virtual void WarmupModel(const std::string& model_name, int warmup) = 0;
  virtual void WarmupDone() = 0;
  virtual void InferBusyLoop(const std::string &model, size_t concurrency, 
                             std::function<double_ms_t(size_t)> interval_fn,
                             double delay_before_infer, int warmup,
                             int64_t show_result = 0) = 0;
  virtual void InferTrace(const std::string &model, size_t concurrency, 
                          const std::vector<double> &start_points, 
                          double delay_before_infer, int warmup,
                          int64_t show_result = 0) = 0;
  virtual void Train(const std::string &model, size_t num_epoch, size_t batch_size) = 0;
  virtual void Report(int verbose = false, std::ostream &os = std::cout) = 0;
};

std::unique_ptr<IWorkload> GetColsysWorkload(std::shared_ptr<grpc::Channel> channel,
           std::chrono::seconds duration, double delay_before_profile,
           const std::string &infer_timeline);

std::unique_ptr<IWorkload> GetTritonWorkload(std::shared_ptr<grpc::Channel> channel,
           std::chrono::seconds duration, double delay_before_profile,
           const std::string &infer_timeline);
}

#endif