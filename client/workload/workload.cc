#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <ratio>
#include <sstream>
#include <thread>
#include <utility>
#include <mutex>

#include "colserve.pb.h"
#include "glog/logging.h"
#include "workload.h"


namespace colserve {
namespace workload {
namespace {
double ComputeThpt(const std::vector<Record> &records) {
  auto first_resp_time = std::min_element(records.begin(), records.end(), [](const Record &a, const Record &b) {
    return a.response_time_ < b.response_time_;
  })->response_time_;
  auto last_request_time = std::max_element(records.begin(), records.end(), [](const Record &a, const Record &b) {
    return a.request_time_ < b.request_time_;
  })->request_time_;
  auto thpt_req_cnt = std::accumulate(records.begin(), records.end(), 0, 
      [first_resp_time, last_request_time] (int acc, const Record &record) {
        return acc + (record.response_time_ >= first_resp_time && record.response_time_ <= last_request_time);
      });
  auto thpt_duration = std::chrono::duration<double>(last_request_time - first_resp_time).count();
  return 1.0 * thpt_req_cnt / thpt_duration;
}
}

void Workload::WarmupModel(const std::string& model_name, int warmup) {
  LOG(INFO) << "Start to send " <<  warmup << " warmup infer request(s) for " << model_name << ".";
  auto set_request_fn = GetSetRequestFn(model_name);
  for(decltype(warmup) k = 0; k < warmup; ++k) {
    grpc::ClientContext context;
    InferResult result;
    InferRequest request;
    set_request_fn(request);
    grpc::Status status = stub_->Inference(&context, request, &result);
    CHECK(status.ok());
  }
  LOG(INFO) << "Complete sending " <<  warmup << " warmup infer request(s) for " << model_name << ".";
}

void Workload::WarmupDone() {
  grpc::ClientContext context;
  EmptyRequest request;
  EmptyResult result;
  grpc::Status status = stub_->WarmupDone(&context, request, &result);
  CHECK(status.ok());
}

void InferWorker::RequestInferBusyLoop(Workload &workload, double delay_before_infer) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << this << ") " << model_ << " BUSY LOOP] ";

  workload.ready_future_.wait();
  std::this_thread::sleep_for(delay_before_infer * std::chrono::seconds(1));

  LOG(INFO) << log_prefix.str() << "RequestInfer start";
  while(workload.running_) {
    // InferRecorder recorder(workload, model_);
    size_t slot = static_cast<size_t>(-1);
    while (slot == static_cast<size_t>(-1)) {
      std::unique_lock status_lock{slot_status_mutex_};
      if (status_slots_id_[InferReqStatus::kReady].empty()) {
        std::shared_lock slot_lock{slot_mutex_};
        for (auto s : status_slots_id_[InferReqStatus::kDone]) {
          if (std::chrono::steady_clock::now() >= slots_[s]->req_status_.ready_time_) {
            slot = s;
            status_slots_id_[InferReqStatus::kDone].erase(slot);
            status_slots_id_[InferReqStatus::kWait].insert(slot);
            break;
          }
        }
      } else {
        auto it = status_slots_id_[InferReqStatus::kReady].begin();
        slot = *it;
        status_slots_id_[InferReqStatus::kReady].erase(it);
        status_slots_id_[InferReqStatus::kWait].insert(slot);
      }
    }
    {
      std::shared_lock slot_lock{slot_mutex_};
      slots_[slot]->req_status_.status_ = InferReqStatus::kWait;
      slots_[slot]->req_status_.request_time_ = std::chrono::steady_clock::now();
      slots_[slot]->req_status_.time_stamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      slots_[slot]->rpc_context_ = std::make_unique<grpc::ClientContext>();
      slots_[slot]->rpc_ = workload.stub_->AsyncInference(slots_[slot]->rpc_context_.get(), slots_[slot]->request_, &cq_);
      slots_[slot]->rpc_->Finish(&slots_[slot]->result_, &slots_[slot]->rpc_status_, (void*)slot);
    }
  }
  LOG(INFO) << log_prefix.str() << "RequestInfer stop";
}

void InferWorker::RequestInferTrace(Workload& workload, 
                                    const std::vector<double>& start_points, 
                                    double delay_before_infer) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << this << ") " << model_ << " TRACE] "; 
  workload.ready_future_.wait();

  LOG(INFO) << log_prefix.str() << "delay " << delay_before_infer << " sec.";
  std::this_thread::sleep_for(delay_before_infer * std::chrono::seconds(1));

  {
    size_t debug_num = std::min(start_points.size(), 10UL);
    std::stringstream debug_stream;
    debug_stream << log_prefix.str() << "RequestInfer start, " 
                 << "len(start_points)=" << start_points.size() 
                 << ", start_points[0:" << debug_num << "]={";
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
    // InferRecorder recorder(workload, model_);
    std::this_thread::sleep_until(sleep_until_sys_clock);
    CHECK(workload.running_) << log_prefix.str() << "Workload client is not running while request(start_point=" << start_point << "s) did not sent.";
    size_t slot;
    {
      std::unique_lock status_lock{slot_status_mutex_};
      if (status_slots_id_[InferReqStatus::kReady].empty()) {
        std::unique_lock slot_lock{slot_mutex_};
        slot = slots_.size();
        auto infer_slot = std::make_shared<InferSlot>();
        set_request_fn_(infer_slot->request_);
        slots_.emplace_back(infer_slot);
        status_slots_id_[InferReqStatus::kWait].insert(slot);
      } else {
        auto it = status_slots_id_[InferReqStatus::kReady].begin();
        slot = *it;
        status_slots_id_[InferReqStatus::kReady].erase(it);
        status_slots_id_[InferReqStatus::kWait].insert(slot);
      }
    }
    {
      std::shared_lock slot_lock{slot_mutex_};
      CHECK_EQ(slots_[slot]->req_status_.status_, InferReqStatus::kReady);
      slots_[slot]->req_status_.status_ = InferReqStatus::kWait;
      slots_[slot]->req_status_.request_time_ = std::chrono::steady_clock::now();
      slots_[slot]->req_status_.time_stamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
      slots_[slot]->rpc_context_ = std::make_unique<grpc::ClientContext>();
      slots_[slot]->rpc_ = workload.stub_->AsyncInference(slots_[slot]->rpc_context_.get(), slots_[slot]->request_, &cq_);
      slots_[slot]->rpc_->Finish(&slots_[slot]->result_, &slots_[slot]->rpc_status_, (void*)slot);
    }
    // for (; i < concurrency_; i++) {
    //   CHECK_NE(request_status_[i].status_, InferReqStatus::kDone);
    //   if (request_status_[i].status_ == InferReqStatus::kReady) {
    //     request_status_[i].status_ = InferReqStatus::kWait;
    //     request_status_[i].request_time_ = std::chrono::steady_clock::now();
    //     contexts_[i] = std::make_unique<grpc::ClientContext>();
    //     rpcs_[i] = workload.stub_->AsyncInference(contexts_[i].get(), requests_[i], &cq_);
    //     rpcs_[i]->Finish(&infer_results_[i], &rpc_status_[i], (void*)i);
    //     break;
    //   }
    // }
    // CHECK_NE(i, concurrency_) << log_prefix.str() << "Unable to find free REQUEST_STATUS, so distribution may be violated.";
  }
  while(workload.running_) {
    DLOG(INFO) << log_prefix.str() << "Workload client is still running, wait 500ms.";
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  LOG(INFO) << log_prefix.str() << "RequestInfer stop";
}

void InferWorker::FetchInferResult(Workload &workload, 
                                   std::function<double_ms_t(size_t)> interval_fn, 
                                   int64_t show_result) {
  std::stringstream log_prefix;
  log_prefix << "[InferWorker(" << std::hex << this << ") " << model_ << "] ";

  workload.ready_future_.wait();
  LOG(INFO) << log_prefix.str() << "FetchInferResult start";
  void *tag;
  bool ok = false;
  bool running = true;
  while(running) {
    // cq_.Next(&tag, &ok);
    while (true) {
      auto next_status = 
          cq_.AsyncNext(&tag, &ok, std::chrono::system_clock::now() + std::chrono::seconds(5));
      if (next_status == grpc::CompletionQueue::GOT_EVENT) {
        break;
      } else if (next_status == grpc::CompletionQueue::SHUTDOWN) {
        running = false;
        break;
      } else { // TIMEOUT
        if (workload.running_) 
          continue;
        {
          std::unique_lock status_lock{slot_status_mutex_};
          running = !status_slots_id_[InferReqStatus::kWait].empty();
        }
        if (!running) break;
      }
    }
    if (!running) break;
  
    auto slot = reinterpret_cast<size_t>(tag);
    // CHECK(rpc_status_[i].ok());
    {
      std::unique_lock status_lock{slot_status_mutex_};
      std::shared_lock slot_lock{slot_mutex_};
      CHECK(slots_[slot]->rpc_status_.ok()) 
          << " slot " << slot << ": " << slots_[slot]->rpc_status_.error_code() << " " 
          << slots_[slot]->rpc_status_.error_message();
      CHECK(status_slots_id_[InferReqStatus::kWait].count(slot)) 
          << " " << slot << " " << status_slots_id_[InferReqStatus::kWait].size();
    }
    auto response_time = std::chrono::steady_clock::now();
    // latency_.push_back(std::chrono::duration<double, std::milli>(end - request_status_[i].request_time_).count());

    {
      std::shared_lock slot_lock{slot_mutex_};
      auto latency = std::chrono::duration<double, std::milli>(response_time - slots_[slot]->req_status_.request_time_).count();
      records_.push_back({slots_[slot]->request_.model(), latency, 
                         slots_[slot]->req_status_.time_stamp_, GetTimeStamp(), 
                         slots_[slot]->req_status_.request_time_, response_time});
      if (show_result > 0) { // check outputs
        std::stringstream ss;
        size_t numel = slots_[slot]->result_.outputs(0).data().size() / sizeof(float);
        ss << "request " << slot << " model " << slots_[slot]->request_.model() << " numel " << numel
          << " result[:" << show_result << "] : ";
        for (size_t j = 0; j < numel && j < show_result; j++) {
          ss << reinterpret_cast<const float*>(slots_[slot]->result_.outputs(0).data().data())[j] << " ";
        }
        ss << "\n";
        std::cout << ss.str();
      } else if (show_result < 0) {
        std::stringstream ss;
        size_t numel = slots_[slot]->result_.outputs(0).data().size() / sizeof(float);
        ss << "request " << slot << " model " << slots_[slot]->request_.model() << " numel " << numel
          << " result[" << show_result << ":] : ";
        size_t j = std::max(0L, static_cast<int64_t>(numel) + show_result);
        for (; j < numel; j++) {
          ss << reinterpret_cast<const float*>(slots_[slot]->result_.outputs(0).data().data())[j] << " ";
        }
        ss << "\n";
        std::cout << ss.str();
      }
    }
    {
      std::unique_lock status_lock{slot_status_mutex_};
      std::shared_lock slot_lock{slot_mutex_};
      if (interval_fn == nullptr) {
        slots_[slot]->req_status_.ready_time_ = std::chrono::steady_clock::now();
        slots_[slot]->req_status_.status_ = InferReqStatus::kReady;
        status_slots_id_[InferReqStatus::kWait].erase(slot);
        status_slots_id_[InferReqStatus::kReady].insert(slot);
      } else {
        slots_[slot]->req_status_.ready_time_ = std::chrono::steady_clock::now() + 
            std::chrono::duration_cast<time_point_t::duration>(interval_fn(slot));
        slots_[slot]->req_status_.status_ = InferReqStatus::kDone;
        status_slots_id_[InferReqStatus::kWait].erase(slot);
        status_slots_id_[InferReqStatus::kDone].insert(slot);
      }
    }
  }
  LOG(INFO) << log_prefix.str() << "FetchInferResult stop";
}

void InferWorker::Report(Workload &workload, int verbose, std::ostream &os) {
  auto records = GetRecord(workload);
  if (records.empty()) {
    os << "[InferWorker TRACE " << model_ << "] no inference record" << std::endl;
    return;
  }
  std::sort(records.begin(), records.end(), [](const Record &a, const Record &b) {
    return a.latency_ < b.latency_;
  });
  std::vector<std::vector<std::string>> table;
  auto f64_to_string = [](double ms) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << ms;
    return ss.str();
  };

  auto ltc_avg = std::accumulate(records.begin(), records.end(), 0, 
      [] (double acc, const Record &record) {
        return acc + record.latency_;
      }) / records.size();
  
  table.push_back({"cnt: " + std::to_string(records.size()), 
                    "ltc.avg: " + f64_to_string(ltc_avg),
                    "ltc.max: " + f64_to_string(records.back().latency_),
                    "thpt: " + f64_to_string(ComputeThpt(records)) + "r/s"});
  for (int i = 99; i > 0;) {
    std::vector<std::string> row;
    for (int j = 0; j < 4 && i > 0; j++) {
      row.push_back("p" + std::to_string(i) + ": " + f64_to_string(records[records.size() * i / 100].latency_));
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
    std::sort(records.begin(), records.end(), [](const Record &a, const Record &b) {
      return a.request_time_ < b.request_time_;
    });
    os << "[InferWorker TRACE " << model_ << "] start_time_stamp, end_time_stamp, request_time(ms), response_time(ms), latency(ms)" << std::endl;
    for (auto &record : records) {
      os << std::fixed << std::setprecision(1)
         << record.start_time_stamp_ << ", " << record.end_time_stamp_ << ", "
         << std::chrono::duration<double, std::milli>(record.request_time_ - workload.run_btime_).count() << ", "
         << std::chrono::duration<double, std::milli>(record.response_time_ - workload.run_btime_).count() << ", "
         << record.latency_ << std::endl;
    }
  }
}

const std::vector<Record> InferWorker::GetRecord(Workload &workload) const {
  if (records_.empty()) {
    return records_;
  }
  if (workload.delay_before_profile_ > 0) {
    std::vector<Record> records;
    for (auto &record : records_) {
      if (record.start_time_stamp_ > workload.start_time_stamp_ + static_cast<long>(workload.delay_before_profile_ * 1000)) {
        records.push_back(record);
      }
    }
    return records;
  } else {
    return records_;
  }
}

void TrainWorker::RequestTrain(Workload &workload) {
  std::stringstream log_prefix;
  log_prefix << "[TrainWorker(" << std::hex << this << ") " << model_ << "] ";

  workload.ready_future_.wait();

  LOG(INFO) << log_prefix.str() << "RequestTrain start";
  while (workload.running_) {
    auto begin = std::chrono::steady_clock::now();
    auto start_time_stamp = GetTimeStamp();
    LOG(INFO) << log_prefix.str() << "RequestTrain send";
    grpc::ClientContext context;
    grpc::Status status = workload.stub_->Train(&context, request_, &train_result_);
    CHECK(status.ok());
    auto end_time_stamp = GetTimeStamp();
    auto end = std::chrono::steady_clock::now();
    // latency_.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    records_.push_back(
        {model_, std::chrono::duration<double, std::milli>(end - begin).count(), start_time_stamp, end_time_stamp, begin, end});
    break;
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
    os << "[TrainWorker TRACE " << model_ << "] start_time_stamp, end_time_stamp, request_time(ms), response_time(ms), latency(ms)" << std::endl;
    for (auto &record : records_) {
      os << std::fixed << std::setprecision(1)
         << record.start_time_stamp_ << ", " << record.end_time_stamp_ << ", "
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

bool Workload::ReportTimeStampToServer() {
  TimeStampRequest request;
  EmptyResult result;
  request.set_time_stamp(start_time_stamp_);
  request.set_delay_before_profile(delay_before_profile_);

  grpc::ClientContext context;
  grpc::CompletionQueue cq;
  grpc::Status status = stub_->ReportStartTimeStamp(&context, request, &result);
  CHECK(status.ok()) << "ReportTimeStampToServer Failed " << status.error_message() << "|" << status.error_details();
  LOG(INFO) << "ReportTimeStampToServer " << start_time_stamp_;
  return true;
}

std::function<void(InferRequest&)> Workload::GetSetRequestFn(const std::string &model) {
  if (model.find("mnist") != std::string::npos) {
    return SetMnistRequestFn(model);
  } else if (model.find("resnet") != std::string::npos
      || model.find("vgg") != std::string::npos
      || model.find("densenet") != std::string::npos
      || model.find("vit") != std::string::npos
      || model.find("swin") != std::string::npos
      || model.find("efficientnet") != std::string::npos
  ) {
    return SetResnetRequestFn(model);
  } else if ( model.find("inception") != std::string::npos) {
    return SetInceptionRequestFn(model);
  } else if (model.find("bert") != std::string::npos) {
    return SetBertRequestFn(model);
  } else if (model.find("gpt") != std::string::npos) {
    return SetGPTRequestFn(model);
  } else {
    LOG(FATAL) << "unable to find SetRequestFn for " << model;
  }
}

std::function<void(InferRequest&)> Workload::SetMnistRequestFn(const std::string &model) {
  static std::mutex mnist_input_datas_mutex;
  static std::vector<std::string> mnist_input_datas;

  {
    std::unique_lock lock{mnist_input_datas_mutex};
    if (mnist_input_datas.empty()) {
      for (size_t i = 0; i < 10; i++) {
        mnist_input_datas.push_back(ReadInput("mnist/input-" + std::to_string(i) + ".bin"));
      }
    }
  }

  auto set_mnist_request_fn = [&](InferRequest &request) {
    static uint32_t i = 0;
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("float32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(28);
    request.mutable_inputs(0)->add_shape(28);
    request.mutable_inputs(0)->set_data(mnist_input_datas[i % 10]);
    i++;
  };
  return set_mnist_request_fn;
}


std::function<void(InferRequest&)> Workload::SetResnetRequestFn(const std::string &model) {
  static std::mutex resnet_input_datas_mutex;
  static std::vector<std::string> resnet_input_datas;

  {
    std::unique_lock lock(resnet_input_datas_mutex);
    if (resnet_input_datas.empty()) {
      for (size_t i = 0; i < 1; i++) {
        resnet_input_datas.push_back(ReadInput("resnet/input-" + std::to_string(i) + ".bin"));
      }
    }
  }

  auto set_resnet_request_fn = [&](InferRequest &request) {
    static uint32_t i = 0;
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("float32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(3);
    request.mutable_inputs(0)->add_shape(224);
    request.mutable_inputs(0)->add_shape(224);
    request.mutable_inputs(0)->set_data(resnet_input_datas[0]);
    i++;
  };
  return set_resnet_request_fn;
}


std::function<void(InferRequest&)> Workload::SetInceptionRequestFn(const std::string &model) {
  static  std::mutex inception_input_datas_mutex;
  static std::vector<std::string> resnet_input_datas;

  {
    std::unique_lock lock{inception_input_datas_mutex};
    if (resnet_input_datas.empty()) {
      for (size_t i = 0; i < 1; i++) {
        resnet_input_datas.push_back(ReadInput("inception/input-" + std::to_string(i) + ".bin"));
      }
    }
  }

  auto set_resnet_request_fn = [&](InferRequest &request) {
    static uint32_t i = 0;
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("float32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(3);
    request.mutable_inputs(0)->add_shape(299);
    request.mutable_inputs(0)->add_shape(299);
    request.mutable_inputs(0)->set_data(resnet_input_datas[0]);
    i++;
  };
  return set_resnet_request_fn;
}

std::function<void(InferRequest&)> Workload::SetBertRequestFn(const std::string &model) {
  static std::mutex bert_input_datas_mutex;
  static std::vector<std::string> bert_input_datas;
  static std::vector<std::string> bert_mask_datas;

  {
    std::unique_lock lock(bert_input_datas_mutex);
    if (bert_input_datas.empty()) {
      for (size_t i = 0; i < 1; i++) {
        bert_input_datas.push_back(ReadInput("bert/input-" + std::to_string(i) + ".bin"));
        bert_mask_datas.push_back(ReadInput("bert/mask-" + std::to_string(i) + ".bin"));
      }
    }
  }
  
  auto set_bert_request_fn = [&](InferRequest &request) {
    static uint32_t i = 0;
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("int64");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(64);
    request.mutable_inputs(0)->set_data(bert_input_datas[0]);
    request.add_inputs();
    request.mutable_inputs(1)->set_dtype("int64");
    request.mutable_inputs(1)->add_shape(1);
    request.mutable_inputs(1)->add_shape(64);
    request.mutable_inputs(1)->set_data(bert_mask_datas[0]);
    i++;
  };
  return set_bert_request_fn;
}

std::function<void(InferRequest&)> SetGPTRequestFn(const std::string &model) {
  static std::mutex gpt_input_datas_mutex;
  static std::vector<std::string> gpt_input_datas;

  {
    std::unique_lock lock(gpt_input_datas_mutex);
    if (gpt_input_datas.empty()) {
      for (size_t i = 0; i < 1; i++) {
        gpt_input_datas.push_back(ReadInput("bert/input-" + std::to_string(i) + ".bin"));
      }
    }
  }
  
  auto set_gpt_request_fn = [&](InferRequest &request) {
    static uint32_t i = 0;
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("int64");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(64);
    request.mutable_inputs(0)->set_data(gpt_input_datas[0]);
    i++;
  };
  return set_gpt_request_fn;
}

void Workload::InferBusyLoop(const std::string &model, size_t concurrency, 
                             std::function<double_ms_t(size_t)> interval_fn,
                             double delay_before_infer, int warmup,
                             int64_t show_result) {
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
      model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInferBusyLoop, worker.get(), std::ref(*this), delay_before_infer));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      interval_fn, show_result));
  infer_workers_.push_back(std::move(worker));
}

void Workload::InferTrace(const std::string &model, size_t concurrency, 
                          const std::vector<double> &start_points, 
                          double delay_before_infer,
                          int warmup, int64_t show_result) {
  auto set_request_fn = GetSetRequestFn(model);
  auto worker = std::make_unique<InferWorker>(
      model, concurrency, set_request_fn, *this);
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::RequestInferTrace, worker.get(), std::ref(*this), start_points, delay_before_infer));
  threads_.push_back(std::make_unique<std::thread>(
      &InferWorker::FetchInferResult, worker.get(), std::ref(*this),
      nullptr, show_result));
  infer_workers_.push_back(std::move(worker));
}

void Workload::Train(const std::string &model, size_t num_epoch, size_t batch_size) {
  auto set_resnet_request_fn = [&](TrainRequest &request) {
    std::stringstream args;
    args << "num-epoch=" << num_epoch << ", batch-size=" << batch_size;
    request.set_model(model);
    request.set_args(args.str());
  };
  
  auto worker = std::make_unique<TrainWorker>(model, set_resnet_request_fn);
  threads_.push_back(std::make_unique<std::thread>(
      &TrainWorker::RequestTrain, worker.get(), std::ref(*this)));
  train_workers_.push_back(std::move(worker));
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
  InferOverallReport(os);
  for (auto &worker : infer_workers_) {
    worker->Report(*this, verbose, os);
  }
  for (auto &worker : train_workers_) {
    worker->Report(*this, verbose, os);
  }
}

void Workload::InferOverallReport(std::ostream &os) {
  std::vector<Record> all_records;
  for (auto &worker : infer_workers_) {
    auto records = worker->GetRecord(*this);
    all_records.insert(all_records.end(), records.begin(), records.end());
  }
  if (all_records.empty()) {
    os << "[Workload TRACE OVERALL] no inference record" << std::endl;
    return;
  }
  { // recored time line
    auto records_copy = all_records;
    std::sort(records_copy.begin(), records_copy.end(), [](const Record &a, const Record &b) {
      return a.start_time_stamp_ < b.start_time_stamp_;
    });
    for (auto &record : records_copy) {
      timeline_handle_ << record.model_name_ << "," << record.start_time_stamp_ << ","  << record.end_time_stamp_ << "\n";
    }
  }
  auto ltc_avg = std::accumulate(all_records.begin(), all_records.end(), 0.0, 
      [](double acc, const Record &record) {
        return acc + record.latency_;
      }) / all_records.size();

  std::sort(all_records.begin(), all_records.end(), [](const Record &a, const Record &b) {
    return a.latency_ < b.latency_;
  });

  auto ltc_min = all_records.front().latency_;
  auto ltc_max = all_records.back().latency_;

  os << "[Infer Overall] start time stamp " << start_time_stamp_ << "\n"
     << "cnt " << all_records.size() 
     << std::fixed << std::setprecision(1)
     << " ltc_avg " << ltc_avg << " ltc min " << ltc_min 
     << " ltc max " << ltc_max << " thpt " << ComputeThpt(all_records) << "\n"
     << "p99 " << all_records[all_records.size() * 99 / 100].latency_ << " "
     << "p95 " << all_records[all_records.size() * 95 / 100].latency_ << " "
     << "p90 " << all_records[all_records.size() * 90 / 100].latency_ << " "
     << "p80 " << all_records[all_records.size() * 80 / 100].latency_ << " "
     << "p70 " << all_records[all_records.size() * 70 / 100].latency_ << " "
     << "p60 " << all_records[all_records.size() * 60 / 100].latency_ << " "
     << "p50 " << all_records[all_records.size() * 50 / 100].latency_ << "\n"
     << std::endl;
}

}  // namespace workload
}