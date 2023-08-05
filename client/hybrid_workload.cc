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

#include "colserve.grpc.pb.h"


std::string ReadInput(const std::string &data_path) {
  std::ifstream data_file{data_path, std::ios::binary};
  CHECK(data_file.good()) << "data " << data_path << " not exist";
  std::string data{std::istreambuf_iterator<char>(data_file), std::istreambuf_iterator<char>()};
  data_file.close();
  return data;
}

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
    begins_.resize(concurrency_);
    status_.resize(concurrency_);
    set_request_fn(requests_);
  }

  void RequestInfer(Workload &workload);
  void FetchInferResult(Workload &workload);
  void Report(std::chrono::seconds duration) {
    std::sort(latency_.begin(), latency_.end());
    std::vector<std::vector<std::string>> table;
    auto f64_to_string = [](double ms) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(1) << ms;
      return ss.str();
    };

    table.push_back({"cnt: " + std::to_string(latency_.size()), 
                     "ltc.avg: " + f64_to_string(std::accumulate(latency_.begin(), latency_.end(), 0.0) / latency_.size()),
                     "ltc.max: " + f64_to_string(latency_.back()),
                     "thpt: " + f64_to_string(1.0 * latency_.size() / duration.count()) + "r/s"});
    for (int i = 99; i > 0;) {
      std::vector<std::string> row;
      for (int j = 0; j < 4 && i > 0; j++) {
        row.push_back("p" + std::to_string(i) + ": " + f64_to_string(latency_[latency_.size() * i / 100]));
        if (i == 99) {
          i -= 4;
        } else {
          i -= 5;
        }
      }
      table.push_back(std::move(row));
    }
    std::vector<size_t> col_width(4);
    for (auto &row : table) {
      for (size_t i = 0; i < row.size(); i++) {
        col_width[i] = std::max(col_width[i], row[i].size());
      }
    }
    auto total_width = std::accumulate(col_width.begin(), col_width.end(), 0) + col_width.size() * 3 + 1;
    std::cout << std::string(total_width, '=') << std::endl;
    auto title = "InferWorker " + model_ + " (" + std::to_string(duration.count()) + " sec)";
    std::cout << '|' << std::string((total_width - 2 - title.size()) / 2, ' ') << title
              << std::string((total_width - 2 - title.size() + 1) / 2, ' ') << '|' << std::endl;
    // std::cout << std::string(total_width, '-') << std::endl;
    for (auto &row : table) {
      std::cout << "| ";
      for (size_t i = 0; i < row.size(); i++) {
        std::cout << std::left << std::setw(col_width[i]) << row[i] << " | ";
      }
      std::cout << std::endl;
    }
    std::cout <<std::string(total_width, '=') << std::endl;
  }

private:
  std::string model_;
  size_t concurrency_;

  grpc::CompletionQueue cq_;
  std::vector<InferRequest> requests_;
  std::vector<InferResult> infer_results_;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts_;
  std::vector<std::unique_ptr<grpc::ClientAsyncResponseReader<InferResult>>> rpcs_;
  std::vector<std::pair<bool, 
                        std::chrono::time_point<std::chrono::steady_clock>>> begins_;
  std::vector<grpc::Status> status_;
  std::vector<double> latency_;
};

class TrainWorker {
 public:
  TrainWorker(const std::string &model, std::function<void(TrainRequest&)> set_request_fn) : model_(model) {
    set_request_fn(request_);  
  }

  void RequestTrain(Workload &workload);
  void Report() {
    auto avg_ltc = std::accumulate(latency_.begin(), latency_.end(), 0) / latency_.size() / 1000;
    std::cout << "[TrainWorker " << model_ << "] "
              << "cnt " << latency_.size() << " | "
              << "ltc.avg " << std::fixed << std::setprecision(2) << avg_ltc << "sec | "
              << "thpt " << std::fixed << std::setprecision(2) << 1.0 / avg_ltc << "job/s"
              << std::endl;
  };
 private:
  std::string model_;

  TrainRequest request_;
  TrainResult train_result_;

  std::vector<double> latency_;
};

class Workload {
 public:
  Workload(std::shared_ptr<grpc::Channel> channel)
      : stub_(ColServe::NewStub(channel)) {
    ready_future_ = std::shared_future<void>{ready_promise_.get_future()};
  };

  bool Hello() {
    EmptyRequest request;
    ServerStatus server_status;

    grpc::ClientContext context;
    grpc::CompletionQueue cq;
    grpc::Status status;

    std::unique_ptr<grpc::ClientAsyncResponseReader<ServerStatus>> rpc(
        stub_->AsyncGetServerStatus(&context, request, &cq));
    rpc->Finish(&server_status, &status, (void*)1);

    void *tag;
    bool ok = false;
    cq.Next(&tag, &ok);
    if (status.ok()) {
      LOG(INFO) << "Server Status: " << server_status.status();
      return true;
    } else {
      LOG(FATAL) << "Query Server Status Failed";
      return false;
    }
  }

  void Run(std::chrono::seconds duration) {
    ready_promise_.set_value();
    running_ = true;
    std::this_thread::sleep_for(duration);
    running_ = false;
    for (auto &thread : threads_) {
      LOG(INFO) << "Worker Thread " << std::hex << thread->get_id() << " joined";
      thread->join();
    }
  }

  void InferMnist(size_t concurrency) {
    static std::vector<std::string> mnist_input_datas;
    if (mnist_input_datas.empty()) {
      for (size_t i = 0; i < 10; i++) {
        mnist_input_datas.push_back(ReadInput("data/mnist/input-" + std::to_string(i) + ".bin"));
      }
    }
    auto set_mnist_request_fn = [&](std::vector<InferRequest> &requests) {
      for (size_t i = 0; i < requests.size(); i++) {
        requests[i].set_model("mnist");
        requests[i].add_inputs();
        requests[i].mutable_inputs(0)->set_dtype("float32");
        requests[i].mutable_inputs(0)->add_shape(1);
        requests[i].mutable_inputs(0)->add_shape(1);
        requests[i].mutable_inputs(0)->add_shape(28);
        requests[i].mutable_inputs(0)->add_shape(28);
        requests[i].mutable_inputs(0)->set_data(mnist_input_datas[i % 10]);
      }
    };

    auto worker = std::make_unique<InferWorker>(
        "mnist", concurrency, set_mnist_request_fn, *this);
    threads_.push_back(std::make_unique<std::thread>(
        &InferWorker::RequestInfer, worker.get(), std::ref(*this)));
    threads_.push_back(std::make_unique<std::thread>(
        &InferWorker::FetchInferResult, worker.get(), std::ref(*this)));
    infer_workers_.push_back(std::move(worker));
  }

  void InferResnet(size_t concurrency) {
    static std::vector<std::string> resnet_input_datas;
    if (resnet_input_datas.empty()) {
      for (size_t i = 0; i < 1; i++) {
        resnet_input_datas.push_back(ReadInput("data/resnet/input-" + std::to_string(i) + ".bin"));
      }
    }

    auto set_resnet_request_fn = [&](std::vector<InferRequest> &requests) {
      for (size_t i = 0; i < requests.size(); i++) {
        requests[i].set_model("resnet152");
        requests[i].add_inputs();
        requests[i].mutable_inputs(0)->set_dtype("float32");
        requests[i].mutable_inputs(0)->add_shape(1);
        requests[i].mutable_inputs(0)->add_shape(3);
        requests[i].mutable_inputs(0)->add_shape(224);
        requests[i].mutable_inputs(0)->add_shape(224);
        requests[i].mutable_inputs(0)->set_data(resnet_input_datas[0]);
      }  
    };

    auto worker = std::make_unique<InferWorker>(
        "resnet152", concurrency, set_resnet_request_fn, *this);
    threads_.push_back(std::make_unique<std::thread>(
        &InferWorker::RequestInfer, worker.get(), std::ref(*this)));
    threads_.push_back(std::make_unique<std::thread>(
        &InferWorker::FetchInferResult, worker.get(), std::ref(*this)));
    infer_workers_.push_back(std::move(worker));
  }
  
  void TrainResnet(size_t num_epoch, size_t batch_size) {
    auto set_resnet_request_fn = [&](TrainRequest &request) {
      std::stringstream args;
      args << "num-epoch=" << num_epoch << ", batch-size=" << batch_size;
      request.set_model("resnet152");
      request.set_args(args.str());
    };
    
    auto worker = std::make_unique<TrainWorker>("resnet152", set_resnet_request_fn);
    threads_.push_back(std::make_unique<std::thread>(
        &TrainWorker::RequestTrain, worker.get(), std::ref(*this)));
    train_workers_.push_back(std::move(worker));
  }

  void Report(std::chrono::seconds duration) {
    for (auto &worker : infer_workers_) {
      worker->Report(duration);
    }
    for (auto &worker : train_workers_) {
      worker->Report();
    }
  }

  friend class InferWorker;
  friend class TrainWorker;
 private:
  std::atomic<bool> running_{false};
  std::promise<void> ready_promise_;
  std::shared_future<void> ready_future_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::vector<std::unique_ptr<InferWorker>> infer_workers_;
  std::vector<std::unique_ptr<TrainWorker>> train_workers_;

  std::unique_ptr<ColServe::Stub> stub_;
};


void InferWorker::RequestInfer(Workload &workload) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << "] ";

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start";
  while(workload.running_) {
    for (size_t i = 0; i < concurrency_; i++) {
      if (begins_[i].first) continue;
      begins_[i].first = true;
      begins_[i].second = std::chrono::steady_clock::now();
      contexts_[i] = std::make_unique<grpc::ClientContext>();
      rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
      rpcs_[i]->Finish(&infer_results_[i], &status_[i], (void*)i);
    }
  }
  LOG(INFO) << log_prefix.str() << "RequestInfer stop";
}

void InferWorker::FetchInferResult(Workload &workload)  {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << "] ";

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "FetchInferResult start";
  void *tag;
  bool ok = false;
  bool running = true;
  while(running) {
    cq_.Next(&tag, &ok);
    size_t i = (size_t)tag;
    CHECK(status_[i].ok());
    auto end = std::chrono::steady_clock::now();
    latency_.push_back(std::chrono::duration<double, std::milli>(end - begins_[i].second).count());
    // { // check outputs
    //   std::stringstream ss;
    //   ss << "request " << i << " result: ";
    //   for (size_t j = 0; j < 10; j++) {
    //     ss << reinterpret_cast<const float*>(infer_results_[i].outputs(0).data().data())[j] << " ";
    //   }
    //   ss << "\n";
    //   std::cout << ss.str();
    // }
    begins_[i].first = false;
    if (workload.running_) continue;
    else {
      running = false;
      for (auto& b : begins_) {
        if (b.first) {
          running = true;
          break;
        }
      }
    }
  }
  LOG(INFO) << log_prefix.str() << "FetchInferResult stop";
}

void TrainWorker::RequestTrain(Workload &workload) {
  std::stringstream log_prefix;
  log_prefix << "[TrainWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << "] ";

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestTrain start";
  while (workload.running_) {
    auto begin = std::chrono::steady_clock::now();
    LOG(INFO) << log_prefix.str() << "RequestTrain send";
    grpc::ClientContext context;
    grpc::Status status = workload.stub_->Train(&context, request_, &train_result_);
    CHECK(status.ok());
    auto end = std::chrono::steady_clock::now();
    latency_.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  LOG(INFO) << log_prefix.str() << "RequestTrain stop";
}


int main(int argc, char** argv) {
  bool enable_train{true}, enable_infer{true};
  std::set<std::string> infer_models, train_models;
  int duration{10}, concurrency{10};
  int num_epoch{1}, batch_size{1};
  CLI::App app{"ColServe Hybrid Workload"};
  app.add_flag("--infer,!--no-infer", enable_infer, "enable infer workload");
  app.add_option("--infer-model", infer_models, "models of infer workload");
  app.add_flag("--train,!--no-train", enable_train, "enable train workload");
  app.add_option("--train-model", train_models, "models of train workload");
  app.add_option("-d,--duration", duration, "duration of workload");
  app.add_option("-c,--concurrency", concurrency, "concurrency of infer workload");
  app.add_option("--num-epoch", num_epoch, "num_epoch of train workload");
  app.add_option("--batch-size", batch_size, "batch_size of train workload");

  CLI11_PARSE(app, argc, argv);

  std::string target = "localhost:8080";
  Workload workload(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
  CHECK(workload.Hello());

  // construct workload
  if (enable_infer) {
    if (infer_models.count("mnist"))
      workload.InferMnist(concurrency);
    if (infer_models.count("resnet"))
      workload.InferResnet(concurrency);
  }
  if (enable_train) {
    if (train_models.count("resnet"))
      workload.TrainResnet(num_epoch, batch_size);
  }

  workload.Run(std::chrono::seconds(duration));

  LOG(INFO) << "report result ...";
  workload.Report(std::chrono::seconds(duration));
  return 0;
}