#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <ratio>
#include <sstream>
#include <thread>
#include <utility>

#include "glog/logging.h"
#include "workload.h"


namespace colserve {
namespace workload {

void InferWorker::RequestInfer(Workload& workload, const std::vector<double>& start_points, double delay_before_infer) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << std::this_thread::get_id() << ") " << model_ << " azure] "; 
  workload.ready_future_.wait();
  std::this_thread::sleep_for(delay_before_infer * std::chrono::seconds(1));
  LOG(INFO) << log_prefix.str() << "Delay " << delay_before_infer << " sec.";
  {
    size_t debug_num = std::min(start_points.size(), 5UL);
    std::stringstream debug_stream;
    debug_stream << log_prefix.str() << "RequestInfer start, " << "len(req_nums)=" << start_points.size() << ", req_nums[0:" << debug_num << "]={";
    for (size_t k=0; k<debug_num; ++k) {
      debug_stream << start_points[k];
      if (k != debug_num - 1) {
        debug_stream << ",";
      }
    }
    debug_stream << "}.";
    LOG(INFO) << debug_stream.str();
  }

  std::mt19937 gen(AppBase::seed);
  auto start_sys_clock = std::chrono::steady_clock::now();
  auto duration_after_start = std::chrono::duration<double>::zero();
  for(auto start_point : start_points) {
    auto sleep_until_sys_clock = start_sys_clock + start_point * std::chrono::seconds(1);
    DLOG(INFO) << log_prefix.str() << "sleep until " << std::chrono::duration_cast<std::chrono::milliseconds>(sleep_until_sys_clock - start_sys_clock).count() << "ms."; 
    std::this_thread::sleep_until(sleep_until_sys_clock);
    size_t i = 0;
    CHECK(workload.running_) << log_prefix.str() << "Workload client is not running while request(start_point=" << start_point << "s) did not sent.";
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
    CHECK_NE(i, concurrency_) << log_prefix.str() << "Unable to find free REQUEST_STATUS, so distribution may be violated.";
  }
  while(workload.running_) {
    DLOG(INFO) << log_prefix.str() << "Workload client is still running, wait 500ms.";
    std::this_thread::sleep_for(500 * std::chrono::milliseconds());
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


void Workload::Infer(const std::string &model, size_t concurrency, const std::vector<double> &start_points, double delay_before_infer,
                            int64_t show_result) {
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
      model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInfer, worker.get(), std::ref(*this), start_points, delay_before_infer));
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

}  // namespace workload
}