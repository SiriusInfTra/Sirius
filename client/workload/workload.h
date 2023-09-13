#ifndef COLSYS_WORKLOAD_H_
#define COLSYS_WORKLOAD_H_

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <future>
#include <map>
#include <pthread.h>
#include <grpcpp/grpcpp.h>
#include <glog/logging.h>
#include <CLI/CLI.hpp>

#include "util.h"
#include "colserve.grpc.pb.h"


namespace colserve {
namespace workload {

using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
using double_ms_t = std::chrono::duration<double, std::milli>;

struct Record {
  double latency_;
  time_point_t request_time_;
  time_point_t response_time_;
};

class Workload;
class InferWorker {
 public:
  InferWorker(const std::string &model, size_t concurrency, 
              std::function<void(std::vector<InferRequest>&)> set_request_fn,
              Workload &workload)
      : model_(model), concurrency_(concurrency){
    requests_.resize(concurrency_);
    infer_results_.resize(concurrency_);
    contexts_.resize(concurrency_);
    rpcs_.resize(concurrency_);
    request_status_.resize(concurrency_);
    rpc_status_.resize(concurrency_);
    set_request_fn(requests_);
  }

  void RequestInfer(Workload &workload);
  void RequestInferPoisson(Workload &workload, double request_per_sec);
  void RequestInferDynamic(Workload &workload, 
                           const std::vector<double> &change_time_points,
                           const std::vector<size_t> &concurrency);
  void FetchInferResult(Workload &workload, 
                        std::function<double_ms_t(size_t)> interval_fn, 
                        uint32_t show_result);
  void Report(Workload &workload, int verbose, std::ostream &os);

private:
  struct InferReqStatus {
    time_point_t request_time_;
    time_point_t ready_time_;
    enum {
      kReady, // ready to send next request
      kWait,  // wait for response
      kDone   // before get ready for next request
    } status_;
  };

  std::string model_;
  size_t concurrency_;

  grpc::CompletionQueue cq_;
  std::vector<InferRequest> requests_;
  std::vector<InferResult> infer_results_;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts_;
  std::vector<std::unique_ptr<grpc::ClientAsyncResponseReader<InferResult>>> rpcs_;
  std::vector<InferReqStatus> request_status_;
  std::vector<grpc::Status> rpc_status_;
  // std::vector<double> latency_;
  std::vector<Record> records_;
};

class TrainWorker {
 public:
  TrainWorker(const std::string &model, std::function<void(TrainRequest&)> set_request_fn) : model_(model) {
    set_request_fn(request_);  
  }

  void RequestTrain(Workload &workload);
  void Report(Workload &workload, int verbose, std::ostream &os);
 private:
  std::string model_;

  TrainRequest request_;
  TrainResult train_result_;

  // std::vector<double> latency_;
  std::vector<Record> records_;
};

class Workload {
 public:
  Workload(std::shared_ptr<grpc::Channel> channel, std::chrono::seconds duration)
      : stub_(ColServe::NewStub(channel)), duration_(duration) {
    ready_future_ = std::shared_future<void>{ready_promise_.get_future()};
  };

  bool Hello();

  void Run() {
    ready_promise_.set_value();
    running_ = true;
    run_btime_ = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(duration_);
    running_ = false;
    for (auto &thread : threads_) {
      LOG(INFO) << "Worker Thread " << std::hex << thread->get_id() << " joined";
      thread->join();
    }
  }

  void InferMnist(size_t concurrency, std::function<double_ms_t(size_t)> interval_fn, 
                  uint32_t show_result = 0);
  void InferMnistPoisson(size_t concurrency, double request_per_sec, 
                         uint32_t show_result = 0);
  void InferMnistDynamic(const std::vector<double> &change_time_points, 
                         const std::vector<size_t> &concurrencys, 
                         uint32_t show_result = 0);
  void InferResnet(const std::string &model, size_t concurrency, std::function<double_ms_t(size_t)> interval_fn, 
                   uint32_t show_result = 0);
  void InferResnetPoisson(size_t concurrency, double request_per_sec, 
                          uint32_t show_result = 0);
  void InferResnetDynamic(const std::vector<double> &change_time_points,
                          const std::vector<size_t> &concurrencys,
                          uint32_t show_result = 0);

  void TrainResnet(size_t num_epoch, size_t batch_size);


  void Report(int verbose = false, std::ostream &os = std::cout);

  friend class InferWorker;
  friend class TrainWorker;
 private:
  std::function<void(std::vector<InferRequest>&)> SetMnistRequestFn();
  std::function<void(std::vector<InferRequest>&)> SetResnetRequestFn(const std::string &model);

  void Infer(const std::string &model, size_t concurrency, 
             std::function<void(std::vector<InferRequest>&)> set_request_fn,
             std::function<double_ms_t(size_t)> interval_fn,
             uint32_t show_result);
  void InferPoisson(const std::string &model, size_t concurrency,
                    std::function<void(std::vector<InferRequest>&)> set_request_fn,
                    double request_per_sec,
                    uint32_t show_result);
  void InferDynamic(const std::string &mode,
                    const std::vector<double> &change_time_points,
                    const std::vector<size_t> &concurrencys,
                    std::function<void(std::vector<InferRequest>&)> set_request_fn,
                    uint32_t show_result);

  std::atomic<bool> running_{false};
  std::promise<void> ready_promise_;
  std::shared_future<void> ready_future_;
  time_point_t run_btime_;
  std::chrono::seconds duration_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::vector<std::unique_ptr<InferWorker>> infer_workers_;
  std::vector<std::unique_ptr<TrainWorker>> train_workers_;

  std::unique_ptr<ColServe::Stub> stub_;
};

}
}

#endif