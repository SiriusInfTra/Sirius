#ifndef COLSYS_WORKLOAD_H_
#define COLSYS_WORKLOAD_H_

#include <climits>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <random>
#include <tuple>
#include <unordered_map>
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
#include <queue>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <memory>
#include <unordered_set>

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
              std::function<void(InferRequest&)> set_request_fn,
              Workload &workload)
      : model_(model), concurrency_(concurrency), set_request_fn_(set_request_fn) {
    for (size_t i = 0; i < concurrency; i++) {
      slots_.emplace_back(std::make_shared<InferSlot>());
      status_slots_id_[InferReqStatus::kReady].insert(i);
    }
    for (auto &slot : slots_) {
      set_request_fn(slot->request_);
    }
  }


  void RequestInferBusyLoop(Workload &workload,
                            double delay_before_infer);
  void RequestInferTrace(Workload& workload, 
                         const std::vector<double>& start_points,
                         double delay_before_infer);
  void FetchInferResult(Workload &workload, 
                        std::function<double_ms_t(size_t)> interval_fn, 
                        int64_t show_result);
  void Report(Workload &workload, int verbose, std::ostream &os);

  const std::vector<Record> & GetRecord() const {
    return records_;
  }

private:
  struct InferReqStatus {
    time_point_t request_time_;
    time_point_t ready_time_;
    enum {
      kReady, // ready to send next request
      kWait,  // wait for response
      kDone,  // before get ready for next request
      kNumStatus
    } status_;
  };

  struct InferSlot {
    InferRequest request_;
    InferResult result_;
    InferReqStatus req_status_;
    grpc::Status rpc_status_;
    std::unique_ptr<grpc::ClientContext> rpc_context_;
    std::unique_ptr<grpc::ClientAsyncResponseReader<InferResult>> rpc_;
  };

  std::string model_;
  size_t concurrency_;

  grpc::CompletionQueue cq_;

  std::mutex slot_status_mutex_;
  std::shared_mutex slot_mutex_;
  std::vector<std::shared_ptr<InferSlot>> slots_;
  std::unordered_set<size_t> status_slots_id_[InferReqStatus::kNumStatus];

  std::function<void(InferRequest&)> set_request_fn_;

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

struct AzureTrace {
  std::vector<double> req_nums;
  double duration_period;
};

class Workload {
 public:
  Workload(std::shared_ptr<grpc::Channel> channel, std::chrono::seconds duration, const std::string &infer_timeline)
      : stub_(ColServe::NewStub(channel)), duration_(duration), timeline_handle_(infer_timeline) {
    ready_future_ = std::shared_future<void>{ready_promise_.get_future()};
    CHECK(timeline_handle_.is_open());
    timeline_handle_ << "model_name,start_time,end_time" << std::endl;
  };

  ~Workload() {
    timeline_handle_.close();
  }

  bool Hello();

  void Run() {
    LOG(INFO) << "Workload start ...";
    running_ = true;
    ready_promise_.set_value();
    run_btime_ = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(duration_);
    LOG(INFO) << "Workload timeout ...";
    running_ = false;
    for (auto &thread : threads_) {
      LOG(INFO) << "Worker Thread " << std::hex << thread->get_id() << " joined";
      thread->join();
    }
  }


  void WarmupModel(const std::string& model_name, int warmup);
  void InferBusyLoop(const std::string &model, size_t concurrency, 
                     std::function<double_ms_t(size_t)> interval_fn,
                     double delay_before_infer, int warmup,
                     int64_t show_result = 0);
  void InferTrace(const std::string &model, size_t concurrency, 
                  const std::vector<double> &start_points, 
                  double delay_before_infer, int warmup,
                  int64_t show_result = 0);
  void TrainResnet(size_t num_epoch, size_t batch_size);


  void Report(int verbose = false, std::ostream &os = std::cout);

  friend class InferWorker;
  friend class TrainWorker;
 private:
  std::function<void(InferRequest&)> GetSetRequestFn(const std::string& model);
  std::function<void(InferRequest&)> SetMnistRequestFn(const std::string &model = "mnist");
  std::function<void(InferRequest&)> SetResnetRequestFn(const std::string &model);
  std::function<void(InferRequest&)> SetBertRequestFn(const std::string &model);

  void InferOverallReport(std::ostream &os);

  std::atomic<bool> running_{false};
  std::promise<void> ready_promise_;
  std::shared_future<void> ready_future_;
  time_point_t run_btime_;
  std::chrono::seconds duration_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::vector<std::unique_ptr<InferWorker>> infer_workers_;
  std::vector<std::unique_ptr<TrainWorker>> train_workers_;

  std::unique_ptr<ColServe::Stub> stub_;

  std::unordered_map<std::string, AzureTrace> azure_model_index_;

  std::mutex timeline_handle_lock_;
  std::ofstream timeline_handle_;

  friend class InferRecorder;

};

class InferRecorder {
public:
  InferRecorder(Workload &workload, const std::string &model_name): model_name_(model_name), workload_(workload) {
    start_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }

  ~InferRecorder() {
    end_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::unique_lock locker{workload_.timeline_handle_lock_};
    workload_.timeline_handle_ << model_name_ << "," << start_ << "," << end_ << "\n";
  }
private:
  const std::string &model_name_;
  Workload &workload_;
  long start_;
  long end_;

};

}
}

#endif