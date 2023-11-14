#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <ratio>
#include <thread>
#include <utility>

#include "glog/logging.h"
#include "workload.h"


namespace colserve {
namespace workload {

void InferWorker::RequestInfer(Workload &workload) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << "] ";

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start";
  while(workload.running_) {
    for (size_t i = 0; i < concurrency_; i++) {
      if (request_status_[i].status_ == InferReqStatus::kDone 
          && std::chrono::steady_clock::now() <= request_status_[i].ready_time_) {
        request_status_[i].status_ = InferReqStatus::kReady;
      }
      if (request_status_[i].status_ == InferReqStatus::kReady) {
        request_status_[i].status_ = InferReqStatus::kWait;
        request_status_[i].request_time_ = std::chrono::steady_clock::now();
        contexts_[i] = std::make_unique<grpc::ClientContext>();
        rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
        rpcs_[i]->Finish(&infer_results_[i], &rpc_status_[i], (void*)i);
      }
    }
  }
  LOG(INFO) << log_prefix.str() << "RequestInfer stop";
}

void InferWorker::RequestInferPoisson(Workload &workload, double request_per_sec) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << " POISSON] ";

  // 10ms as basic unit
  // if server cannot serve requests required by poisson dist, 
  // it will fallback to normal busy requesting
  double req_per_10ms = 1.0 * request_per_sec / 100;
  std::mt19937 gen(AppBase::seed);
  std::poisson_distribution<> dist(req_per_10ms);

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start";
  while (workload.running_) {
    auto begin = std::chrono::steady_clock::now();
    auto num_req = dist(gen);
    for (size_t i = 0; i < concurrency_ && num_req > 0; i++) {
      CHECK_NE(request_status_[i].status_, InferReqStatus::kDone);
      if (request_status_[i].status_ == InferReqStatus::kReady) {
        request_status_[i].status_ = InferReqStatus::kWait;
        request_status_[i].request_time_ = std::chrono::steady_clock::now();
        contexts_[i] = std::make_unique<grpc::ClientContext>();
        rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
        rpcs_[i]->Finish(&infer_results_[i], &rpc_status_[i], (void*)i);
        num_req--;
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = end - begin;
    if (duration < std::chrono::milliseconds(10)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10) - duration);
    }
  }
}

void InferWorker::RequestInferDynamicPoisson(
    Workload &workload,
    std::vector<double> change_time_points,
    std::vector<double> lambdas) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << " DYNAMIC POISSON] ";
  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start";

  std::mt19937 gen(AppBase::seed);

  auto begin = std::chrono::steady_clock::now();\
  size_t idx = 0;
  size_t req_per_10ms = lambdas[idx] / 100;
  std::poisson_distribution<> dist(req_per_10ms);
  while (workload.running_) {
    auto b = std::chrono::steady_clock::now();
    auto d = std::chrono::duration<double>(b-begin).count();
    if (idx < change_time_points.size()) {
      if (d >= change_time_points[idx]) {
        LOG(INFO) << log_prefix.str() << "change poisson lamda " << lambdas[idx] << " -> " << lambdas[idx+1];
        dist = std::poisson_distribution<>(lambdas[++idx] / 100);
      }
    }
    auto num_req = dist(gen);
    for (size_t i = 0; i < concurrency_ && num_req > 0; i++) {
      CHECK_NE(request_status_[i].status_, InferReqStatus::kDone);
      if (request_status_[i].status_ == InferReqStatus::kReady) {
        request_status_[i].status_ = InferReqStatus::kWait;
        request_status_[i].request_time_ = std::chrono::steady_clock::now();
        contexts_[i] = std::make_unique<grpc::ClientContext>();
        rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
        rpcs_[i]->Finish(&infer_results_[i], &rpc_status_[i], (void*)i);
        num_req--;
      }
    }
    auto e = std::chrono::steady_clock::now();
    auto duration = e - b;
    if (duration < std::chrono::milliseconds(10)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10) - duration);
    }
  }

}

void InferWorker::RequestInferDynamic(Workload &workload,
                                      std::vector<double> change_time_points,
                                      std::vector<size_t> concurrencys) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << " DYNAMIC] "; 
  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start, max_concurrency " << concurrency_ << " init_concurrency " << concurrencys[0];

  auto begin = std::chrono::steady_clock::now();
  size_t idx = 0;
  size_t concurrency = concurrencys[0];
  while (workload.running_) {
    auto t = std::chrono::steady_clock::now();
    auto d = std::chrono::duration<double>(t-begin).count();
    if (idx < change_time_points.size()) {
      if (d >= change_time_points[idx]) {
        LOG(INFO) << log_prefix.str() << "change concurrency " << concurrency << " -> " << concurrencys[idx+1];
        concurrency = concurrencys[++idx];
      }
    }
    for (size_t i = 0; i < concurrency; i++) {
      if (request_status_[i].status_ == InferReqStatus::kDone 
          && std::chrono::steady_clock::now() <= request_status_[i].ready_time_) {
        request_status_[i].status_ = InferReqStatus::kReady;
      }
      if (request_status_[i].status_ == InferReqStatus::kReady) {
        request_status_[i].status_ = InferReqStatus::kWait;
        request_status_[i].request_time_ = std::chrono::steady_clock::now();
        contexts_[i] = std::make_unique<grpc::ClientContext>();
        rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
        rpcs_[i]->Finish(&infer_results_[i], &rpc_status_[i], (void*)i);
      }
    }
  }
  LOG(INFO) << log_prefix.str() << "RequestInfer stop";
}

void InferWorker::RequestInferAzure(Workload &workload, const std::vector<double> &req_nums, double period_duration) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << " azure] "; 
  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start, req_nums[0:3]={" 
            << req_nums[0] << "," << req_nums[1] << "," << req_nums[2] << "},"
            << " max_req_num=" << *std::max_element(req_nums.cbegin(), req_nums.cend());

  std::mt19937 gen(AppBase::seed);
  // std::poisson_distribution<> dist(req_per_10ms);

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "RequestInfer start";
  auto start = std::chrono::steady_clock::now();
  auto total_sleep = std::chrono::duration<double>::zero();
  while(workload.running_) {
    size_t minute_id = std::chrono::duration_cast<std::chrono::seconds>((std::chrono::steady_clock::now()- start) / period_duration).count();
    if (minute_id >= req_nums.size()) {
      LOG(WARNING) << "workload still running but minute_id >= period: " << minute_id << " vs " <<  req_nums.size() << ".";
      break;
    }
    double req_per_s = req_nums[minute_id] / 60;
    std::exponential_distribution<> dist(req_per_s);
    double sleep_for = dist(gen);
    auto total_sleep = std::chrono::seconds(1) * sleep_for;
    LOG(INFO) << "sleep_for=" << sleep_for << ", total_sleep=" << total_sleep.count() << ".";
    if (total_sleep > std::chrono::seconds(req_nums.size()) * period_duration) {
      LOG(INFO) << "sleep to long, break.";
      break;
    }
    auto sleep_until = start + total_sleep;
    std::this_thread::sleep_until(sleep_until);
    size_t i = 0;
    for (; i < concurrency_; i++) {
      CHECK_NE(request_status_[i].status_, InferReqStatus::kDone);
      if (request_status_[i].status_ == InferReqStatus::kReady) {
        request_status_[i].status_ = InferReqStatus::kWait;
        request_status_[i].request_time_ = std::chrono::steady_clock::now();
        contexts_[i] = std::make_unique<grpc::ClientContext>();
        rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
        rpcs_[i]->Finish(&infer_results_[i], &rpc_status_[i], (void*)i);
        break;
      }
    }
    CHECK_NE(i, concurrency_);
  }
  LOG(INFO) << log_prefix.str() << "RequestInfer stop";
}

void InferWorker::FetchInferResult(Workload &workload, 
                                   std::function<double_ms_t(size_t)> interval_fn, 
                                   int64_t show_result) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << "] ";

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "FetchInferResult start";
  void *tag;
  bool ok = false;
  bool running = true;
  while(running) {
    // cq_.Next(&tag, &ok);
    while (true) {
      auto next_status = 
          cq_.AsyncNext(&tag, &ok, std::chrono::system_clock::now() + std::chrono::seconds(1));
      if (next_status == grpc::CompletionQueue::GOT_EVENT) {
        break;
      } else if (next_status == grpc::CompletionQueue::SHUTDOWN) {
        running = false;
        break;
      } else { // TIMEOUT
        if (workload.running_) 
          continue;
        running = false;
        for (auto& rs : request_status_) {
          if (rs.status_ == InferReqStatus::kWait) {
            running = true; break;
          }
        }
        if (!running) break;
      }
    }
    if (!running) break;
  
    size_t i = (size_t)tag;
    CHECK(rpc_status_[i].ok());
    auto response_time = std::chrono::steady_clock::now();
    // latency_.push_back(std::chrono::duration<double, std::milli>(end - request_status_[i].request_time_).count());
    auto latency = std::chrono::duration<double, std::milli>(response_time - request_status_[i].request_time_).count();
    records_.push_back({latency, request_status_[i].request_time_, response_time});
    if (show_result > 0) { // check outputs
      std::stringstream ss;
      size_t numel = infer_results_[i].outputs(0).data().size() / sizeof(float);
      ss << "request " << i << " numel " << numel
         << " result[:" << show_result << "] : ";
      for (size_t j = 0; j < numel && j < show_result; j++) {
        ss << reinterpret_cast<const float*>(infer_results_[i].outputs(0).data().data())[j] << " ";
      }
      ss << "\n";
      std::cout << ss.str();
    } else if (show_result < 0) {
      std::stringstream ss;
      size_t numel = infer_results_[i].outputs(0).data().size() / sizeof(float);
      ss << "request " << i << " numel " << numel
         << " result[" << show_result << ":] : ";
      size_t j = std::max(0L, static_cast<int64_t>(numel) + show_result);
      for (; j < numel; j++) {
        ss << reinterpret_cast<const float*>(infer_results_[i].outputs(0).data().data())[j] << " ";
      }
      ss << "\n";
      std::cout << ss.str();
    }
    if (interval_fn == nullptr) {
      request_status_[i].ready_time_ = std::chrono::steady_clock::now();
      request_status_[i].status_ = InferReqStatus::kReady;
    } else {
      request_status_[i].ready_time_ = std::chrono::steady_clock::now() + 
          std::chrono::duration_cast<time_point_t::duration>(interval_fn(i));
      request_status_[i].status_ = InferReqStatus::kDone;
    }
  }
  LOG(INFO) << log_prefix.str() << "FetchInferResult stop";
}

void InferWorker::Report(Workload &workload, int verbose, std::ostream &os) {
  if (records_.empty()) {
    os << "[InferWorker TRACE " << model_ << "] no inference record" << std::endl;
    return;
  }
  std::sort(records_.begin(), records_.end(), [](const Record &a, const Record &b) {
    return a.latency_ < b.latency_;
  });
  std::vector<std::vector<std::string>> table;
  auto f64_to_string = [](double ms) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << ms;
    return ss.str();
  };

  auto ltc_avg = std::accumulate(records_.begin(), records_.end(), 0, 
      [] (double acc, const Record &record) {
        return acc + record.latency_;
      }) / records_.size();
  table.push_back({"cnt: " + std::to_string(records_.size()), 
                    "ltc.avg: " + f64_to_string(ltc_avg),
                    "ltc.max: " + f64_to_string(records_.back().latency_),
                    "thpt: " + f64_to_string(1.0 * records_.size() / workload.duration_.count()) + "r/s"});
  for (int i = 99; i > 0;) {
    std::vector<std::string> row;
    for (int j = 0; j < 4 && i > 0; j++) {
      row.push_back("p" + std::to_string(i) + ": " + f64_to_string(records_[records_.size() * i / 100].latency_));
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
  os << std::string(total_width, '=') << std::endl;
  auto title = "InferWorker " + model_ + " (" + std::to_string(workload.duration_.count()) + " sec)";
  os << '|' << std::string((total_width - 2 - title.size()) / 2, ' ') << title
            << std::string((total_width - 2 - title.size() + 1) / 2, ' ') << '|' << std::endl;
  // std::cout << std::string(total_width, '-') << std::endl;
  for (auto &row : table) {
    os << "| ";
    for (size_t i = 0; i < row.size(); i++) {
      os << std::left << std::setw(col_width[i]) << row[i] << " | ";
    }
    os << std::endl;
  }
  os << std::string(total_width, '=') << std::endl;

  if (verbose) {
    std::sort(records_.begin(), records_.end(), [](const Record &a, const Record &b) {
      return a.request_time_ < b.request_time_;
    });
    os << "[InferWorker TRACE " << model_ << "] request_time(ms), response_time(ms), latency(ms)" << std::endl;
    for (auto &record : records_) {
      os << std::fixed << std::setprecision(1)
         << std::chrono::duration<double, std::milli>(record.request_time_ - workload.run_btime_).count() << ", "
         << std::chrono::duration<double, std::milli>(record.response_time_ - workload.run_btime_).count() << ", "
         << record.latency_ << std::endl;
    }
  }
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
    // latency_.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    records_.push_back(
        {std::chrono::duration<double, std::milli>(end - begin).count(), begin, end});
  }
  LOG(INFO) << log_prefix.str() << "RequestTrain stop";
}

void TrainWorker::Report(Workload &workload, int verbose, std::ostream &os) {
  if (records_.empty()) {
    os << "[TrainWorker " << model_ << "] no train record" << std::endl;
    return;
  }
  auto avg_ltc = std::accumulate(records_.begin(), records_.end(), 0.0, 
      [](double acc, const Record &record) {
        return acc + record.latency_;
      }) / records_.size() / 1000;
  os << "[TrainWorker " << model_ << "] "
     << "cnt " << records_.size() << " | "
     << "ltc.avg " << std::fixed << std::setprecision(2) << avg_ltc << "sec | "
     << "thpt " << std::fixed << std::setprecision(2) << 1.0 / avg_ltc << "job/s"
     << std::endl;
    
  if (verbose) {
    std::sort(records_.begin(), records_.end(), [](const Record &a, const Record &b) {
      return a.request_time_ < b.request_time_;
    });
    os << "[TrainWorker TRACE " << model_ << "] request_time(ms), response_time(ms), latency(ms)" << std::endl;
    for (auto &record : records_) {
      os << std::fixed << std::setprecision(1)
         << std::chrono::duration<double, std::milli>(record.request_time_ - workload.run_btime_).count() << ", "
         << std::chrono::duration<double, std::milli>(record.response_time_ - workload.run_btime_).count() << ", "
         << record.latency_ << std::endl;
    }
  }
}

bool Workload::Hello() {
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
    LOG(FATAL) << "Query Server Status Failed " << status.error_message() << "|" << status.error_details();
    return false;
  }
}

std::function<void(std::vector<InferRequest>&)> Workload::GetSetRequestFn(const std::string &model) {
  if (model.find("mnist") != std::string::npos) {
    return SetMnistRequestFn(model);
  } else if (model.find("resnet") != std::string::npos) {
    return SetResnetRequestFn(model);
  } else {
    LOG(FATAL) << "unable to find SetRequestFn for " << model;
  }
}

std::function<void(std::vector<InferRequest>&)> Workload::SetMnistRequestFn(const std::string &model) {
  static std::vector<std::string> mnist_input_datas;
  if (mnist_input_datas.empty()) {
    for (size_t i = 0; i < 10; i++) {
      mnist_input_datas.push_back(ReadInput("data/mnist/input-" + std::to_string(i) + ".bin"));
    }
  }
  auto set_mnist_request_fn = [&](std::vector<InferRequest> &requests) {
    for (size_t i = 0; i < requests.size(); i++) {
      requests[i].set_model(model);
      requests[i].add_inputs();
      requests[i].mutable_inputs(0)->set_dtype("float32");
      requests[i].mutable_inputs(0)->add_shape(1);
      requests[i].mutable_inputs(0)->add_shape(1);
      requests[i].mutable_inputs(0)->add_shape(28);
      requests[i].mutable_inputs(0)->add_shape(28);
      requests[i].mutable_inputs(0)->set_data(mnist_input_datas[i % 10]);
    }
  };
  return set_mnist_request_fn;
}

// void Workload::InferMnist(size_t concurrency, std::function<double_ms_t(size_t)> interval_fn, 
//                           int64_t show_result) {
//   Infer("mnist", concurrency, SetMnistRequestFn(), interval_fn, show_result);
// }

// void Workload::InferMnistPoisson(size_t concurrency, double request_per_sec,
//                                  int64_t show_result) {
//   InferPoisson("mnist", concurrency, SetMnistRequestFn(), request_per_sec, show_result);
// }

// void Workload::InferMnistDynamic(const std::vector<double> &change_time_points,
//                                  const std::vector<size_t> &concurrencys,
//                                  int64_t show_result) {
//   InferDynamic("mnist", change_time_points, concurrencys, SetMnistRequestFn(), show_result);
// } 


std::function<void(std::vector<InferRequest>&)> Workload::SetResnetRequestFn(const std::string &model) {
  static std::vector<std::string> resnet_input_datas;
  if (resnet_input_datas.empty()) {
    for (size_t i = 0; i < 1; i++) {
      resnet_input_datas.push_back(ReadInput("data/resnet/input-" + std::to_string(i) + ".bin"));
    }
  }

  auto set_resnet_request_fn = [&](std::vector<InferRequest> &requests) {
    for (size_t i = 0; i < requests.size(); i++) {
      requests[i].set_model(model);
      requests[i].add_inputs();
      requests[i].mutable_inputs(0)->set_dtype("float32");
      requests[i].mutable_inputs(0)->add_shape(1);
      requests[i].mutable_inputs(0)->add_shape(3);
      requests[i].mutable_inputs(0)->add_shape(224);
      requests[i].mutable_inputs(0)->add_shape(224);
      requests[i].mutable_inputs(0)->set_data(resnet_input_datas[0]);
    }  
  };
  return set_resnet_request_fn;
}

// void Workload::InferResnet(const std::string &model, size_t concurrency, std::function<double_ms_t(size_t)> interval_fn,
//                            int64_t show_result) {
//   Infer("resnet152", concurrency, SetResnetRequestFn(model), interval_fn, show_result);
// }

// void Workload::InferResnetPoisson(size_t concurrency, double request_per_sec,
//                                   int64_t show_result) {
//   InferPoisson("resnet152", concurrency, SetResnetRequestFn("resnet152"), request_per_sec, show_result);
// }

// void Workload::InferResnetDynamicPoisson(
//     size_t concurrency,
//     const std::vector<double> &change_time_points,
//     const std::vector<double> &lambdas,
//     int64_t show_result) {
//   InferDynamicPoisson("resnet152", concurrency, change_time_points, lambdas,
//       SetResnetRequestFn("resnet152"), show_result);
// } 

// void Workload::InferResnetDynamic(const std::vector<double> &change_time_points,
//                                   const std::vector<size_t> &concurrencys,
//                                   int64_t show_result) {
//   InferDynamic("resnet152", change_time_points, concurrencys, SetResnetRequestFn("resnet152"), show_result);
// }

void Workload::Infer(const std::string &model, size_t concurrency, 
                    //  std::function<void(std::vector<InferRequest>&)> set_request_fn,
                     std::function<double_ms_t(size_t)> interval_fn,
                     int64_t show_result) {
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
      model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInfer, worker.get(), std::ref(*this)));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      interval_fn, show_result));
  infer_workers_.push_back(std::move(worker));
}


void Workload::InferPoisson(const std::string &model, size_t concurrency,
                            // std::function<void(std::vector<InferRequest>&)> set_request_fn,
                            double request_per_sec,
                            int64_t show_result) {
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
      model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInferPoisson, worker.get(), std::ref(*this), 
      request_per_sec));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      nullptr, show_result));
  infer_workers_.push_back(std::move(worker));
}

void Workload::InferDynamicPoisson(
    const std::string &model,
    size_t concurrency,
    const std::vector<double> &change_time_points,
    const std::vector<double> &lambdas,
    // std::function<void(std::vector<InferRequest>&)> set_request_fn,
    int64_t show_result) {
  CHECK_EQ(change_time_points.size() + 1, lambdas.size()) << model;
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
    model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInferDynamicPoisson,  worker.get(), std::ref(*this),
      change_time_points, lambdas));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      nullptr, show_result));
  infer_workers_.push_back(std::move(worker));
}

void Workload::InferDynamic(const std::string &model,
                            const std::vector<double> &change_time_points,
                            const std::vector<size_t> &concurrencys,
                            // std::function<void(std::vector<InferRequest>&)> set_request_fn,
                            int64_t show_result) {
  CHECK_EQ(change_time_points.size() + 1, concurrencys.size()) << model;
  auto set_request_fn = GetSetRequestFn(model);
  auto max_concurrency = *std::max_element(concurrencys.begin(), concurrencys.end());
  auto worker = std::make_unique<InferWorker>(
    model, max_concurrency, set_request_fn, *this);
  // LOG(INFO) << concurrencys[0];
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInferDynamic, worker.get(), std::ref(*this),
      change_time_points, concurrencys));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      nullptr, show_result));
  infer_workers_.push_back(std::move(worker));
}

void Workload::InferAzure(const std::string &model, unsigned model_num, 
                          const std::vector<std::vector<unsigned>> &trace_data, 
                          double scale_factor, double period_duration, 
                          size_t concurrency, int64_t show_result) {
  std::vector<double> req_nums(trace_data.front().size());
  for(size_t minute_id = 0; minute_id < req_nums.size(); ++minute_id) {
    unsigned counter = 0U;
    for(size_t func_id = azure_model_index_.size(); func_id < trace_data.size(); func_id += model_num) {
      counter += trace_data[func_id][minute_id];
    }
    req_nums[minute_id] = static_cast<double>(counter) * scale_factor;
  }
  azure_model_index_.emplace(std::make_pair(model, colserve::workload::AzureTrace{req_nums, period_duration}));
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
    model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInferAzure, worker.get(), std::ref(*this),
      req_nums, period_duration));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      nullptr, show_result));
  infer_workers_.push_back(std::move(worker));
}

void Workload::TrainResnet(size_t num_epoch, size_t batch_size) {
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

void Workload::Report(int verbose, std::ostream &os) {
  for (auto &worker : infer_workers_) {
    worker->Report(*this, verbose, os);
  }
  for (auto &worker : train_workers_) {
    worker->Report(*this, verbose, os);
  }
}

} 
}