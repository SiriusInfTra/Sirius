#include <server/infer_model_store.h>
#include <server/infer_model.h>
#include <server/controller.h>
#include <server/config.h>


namespace colserve {

std::array<std::atomic<int>, static_cast<size_t>(Model::Status::kNumStatus)> 
Model::model_stat_{0, 0, 0};

Model::Model(const std::string &name, const std::filesystem::path &model_path,
             std::optional<const std::map<std::string, tvm::TVMArray>> params,
             DLDevice device, size_t batch_size, size_t num_worker, size_t max_num_worker)
    : name_(name), device_(device), batch_size_(batch_size), max_num_worker_(max_num_worker) {
  // DLOG(INFO) << "Model " << name << " start initializing from " << model_path;
  CHECK(std::filesystem::exists(model_path / "mod.so"));
  CHECK(std::filesystem::exists(model_path / "mod.json"));
  CHECK(std::filesystem::exists(model_path / "mod.params"));

  // rmod_ = tvm::runtime::Module::LoadFromFile((model_path / "mod.so").c_str(), "so");
  auto rmod = 
      ::tvm::runtime::Module::LoadFromFile((model_path / "mod.so").c_str(), "so");

  if (!params.has_value()) {
    tvm_graph_ = std::make_unique<tvm::TVMGraph>(
      InferModelStore::GetModelRank(),
      this,
      name,
      (model_path / "mod.json").c_str(),
      rmod,
      (model_path / "mod.params").c_str(),
      std::vector{device}
    );
  } else {
    tvm_graph_ = std::make_unique<tvm::TVMGraph>(
      InferModelStore::GetModelRank(),
      this,
      name,
      (model_path / "mod.json").c_str(),
      rmod,
      params.value(),
      std::vector{device}
    );
  }

  InitMetaInfo();

  if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    // tvm_graph_->ResetParamStorage();
    LOG(INFO) << "[Model]: " << name << " task switch by pipelining load/run";
  }

  // config infer scaling
  // scale_up_queue_time_ = 200; // no used
  // scale_down_idle_time_ = Config::infer_model_max_idle_ms;
  // warmup_ = Config::has_warmup;
  // max_num_worker_ = 5;
  // infer_workers_.resize(max_num_worker_);
  // worker_running_.resize(max_num_worker_);
  // waited_trains_.resize(max_num_worker_);

  CHECK_LE(num_worker, max_num_worker);
  CHECK_EQ(max_num_worker_, 1) << "currently, only support one worker";
  CHECK_LT(max_num_worker_, MAX_NUM_WORKER) << "max num worker exceed limit";
  for (size_t i = 0; i < num_worker; i++) {
    auto executor = tvm_graph_->CreateGraphExecutor(i, std::vector{device});
    executor->Init(true);
    executors_.push_back(std::move(executor));
    status_.push_back(Status::kReady);
    model_stat_[static_cast<size_t>(Status::kReady)].fetch_add(1, std::memory_order_relaxed);
  }
  infer_workers_.resize(max_num_worker_);

  LOG_IF(INFO, Config::log_model_init_info)
      << "[Model Init] " << name << " initilized " << num_worker << " executor";

  // if (Config::colocate_config.skip_malloc || Config::colocate_config.skip_loading) {
  //   for (size_t i = 0; i < max_num_worker_; i++) {
  //     executors_[i].get()->FakeInit(Config::colocate_config.skip_malloc, Config::colocate_config.skip_loading);
  //   }
  // }
  if (Config::colocate_config.skip_malloc || Config::colocate_config.skip_loading) {
    LOG(FATAL) << "not support skip_malloc or skip_loading";
  }

  for (size_t i = 0; i < num_worker; i++) {
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, 2);
    // infer_workers_.push_back(std::make_unique<std::thread>(&Model::Inference, this, &barrier));
    infer_workers_[i].reset(new std::thread{&Model::Inference, this, i, &barrier});
    // if (Config::IsSwitchMode()) pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
  }

  // if (Config::IsColocateMode() || Config::IsSwitchMode()) {
  //   job_monitor_.reset(new std::thread{&Model::MonitorJob, this});
  // }
}

void Model::InitMetaInfo() {
  auto [shape_info, dtype_info] = 
      tvm_graph_->GetInputInfo();
  for (const auto &kv : shape_info) {
    auto shape = shape_info[kv.first];
    auto dtype = dtype_info[kv.first];
    input_info_[kv.first] = std::make_pair(shape, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); i++) 
      ss << shape[i] << " ";
    LOG_IF(INFO, Config::log_model_init_info) 
        << "[Model Init] Input: " << name_ << " " << kv.first 
        << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape[0], batch_size_) << "batch size mismatch";
  }

  std::tie(shape_info, dtype_info) = tvm_graph_->GetOutputInfo();
  for (const auto &kv : shape_info) {
    auto shape = shape_info[kv.first];
    auto dtype = dtype_info[kv.first];
    output_info_[kv.first] = std::make_pair(shape, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); i++)
      ss << shape[i] << " ";
    LOG_IF(INFO, Config::log_model_init_info) 
        << "[Model Init] Output: " << name_ << " " << kv.first 
        << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape[0], batch_size_) << "batch size mismatch";
  }
}

bool Model::AddJob(network::InferHandler::InferData* data) {
  // LOG(INFO) << "model " << name_ << " add job";
  if (name_ == "dummy") {
    data->GetResponse().set_result("dummy result");
    data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    return true;
  }
  Controller::Get()->InferRequestInc();
  // InterruptTrain check whether to interrupt train
  Controller::Get()->InterruptTrain(); 
  return job_queue_.Put(std::make_shared<InferJob>(data));
}

bool Model::ReclaimMemory(size_t rank) {
  std::unique_lock lock{muts_[rank]};
  if (status_[rank] == Status::kWithoutMemory) {
    return false; 
  }
  executors_[rank]->DeInit();
  status_[rank] = Status::kWithoutMemory;
  model_stat_[static_cast<size_t>(Status::kReady)].fetch_sub(1, std::memory_order_relaxed);
  model_stat_[static_cast<size_t>(Status::kWithoutMemory)].fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool Model::SetupMemory(size_t rank) {
  CHECK(status_[rank] == Status::kWithoutMemory);
  executors_[rank]->Init(false);
  model_stat_[static_cast<size_t>(Status::kWithoutMemory)].fetch_sub(1, std::memory_order_relaxed);
  model_stat_[static_cast<size_t>(Status::kWithoutParam)].fetch_add(1, std::memory_order_relaxed);
  status_[rank] = Status::kWithoutParam;
  return true;
}

bool Model::Inference(uint32_t rank, pthread_barrier_t* barrier) {
  // if (Config::IsSwitchMode()) {
  //   InferModelStore::Get()->TaskSwitchEnter();
  //   CHECK(barrier != nullptr);
  //   pthread_barrier_wait(barrier);
  // }
  // auto graph_executor = tvm_graph_->CreateGraphExecutor();
  auto graph_executor = executors_[rank].get();
  num_worker_.fetch_add(1, std::memory_order_relaxed);
  LOG(INFO) << "[Model Inference] " << name_ << " (rank " << rank << ") start inference";
  if (barrier != nullptr) pthread_barrier_wait(barrier);

  // bool first_exec = true;
  
  // auto last_get_batch_time = std::chrono::steady_clock::now();
  auto last_infer_time = Profiler::Now();
  while (true) {{                  
    // if (Config::IsColocateMode() && Profiler::MilliFrom(last_get_batch_time) >= GetMaxIdleMill()) {
    //   uint32_t num_worker = num_worker_;
    //   if (num_worker - 0 > 0) { /* check if num_worker reduce */
    //     // LOG(INFO) << "num_worker " << num_worker;
    //     auto ok = num_worker_.compare_exchange_strong(num_worker, num_worker - 1,
    //         std::memory_order_relaxed);
    //     if (ok) break;
    //   }
    // } else if (Config::IsSwitchMode() && InferModelStore::Get()->TaskSwitchControlCnter() == 
    //     static_cast<int>(InferModelStore::TaskSwitchStatus::kPrepareExit)) {
    //   // wait all infer enter task switch prepare exit stage
    //   pthread_barrier_wait(&InferModelStore::Get()->task_switch_barrier);
    //   // wait task switch result
    //   pthread_barrier_wait(&InferModelStore::Get()->task_switch_barrier);
    //   if (InferModelStore::Get()->TaskSwitchControlCnter() == 
    //       static_cast<int>(InferModelStore::TaskSwitchStatus::kCancelExit)) { // not exit
    //     // do cancel exit        
    //     pthread_barrier_wait(&InferModelStore::Get()->task_switch_barrier);
    //   } else { // exit
    //     num_worker_.fetch_sub(1, std::memory_order_relaxed);
    //     break;
    //   }
    // }

    // TODO dynamic batching
    auto jobs = job_queue_.GetBatch(batch_size_, 10, 10);
    if (jobs.empty()) {
      auto idle_mill = Profiler::MilliFrom(last_infer_time);
      infer_idle_mills_[rank].store(idle_mill, std::memory_order_relaxed);
      continue;
    }

    // [switch mode] before infering, first claim infering execution
    InferModelStore::InferingInc(executors_[rank].get());

    std::unique_lock lock{muts_[rank]};
    last_infer_time = Profiler::Now();
    infer_idle_mills_[rank].store(0, std::memory_order_relaxed);
    InferModelStore::UpdateLastInferTime();

    double infer_alloc_ms = 0;
    {
      if (status_[rank] == Status::kWithoutMemory) {
        auto begin = Profiler::Now();
        SetupMemory(rank);
        infer_alloc_ms = Profiler::MilliFrom(begin);
      }
    }

    // last_infer_times_[rank] = Profiler::Now();
    DLOG(INFO) << "[Model Inference] GetBatch " << jobs.size() << "/" << batch_size_;

    bool err = false;
    auto infer_begin = std::chrono::steady_clock::now();

    double set_input_ms;
    {
      size_t idx = 0;
      auto begin = std::chrono::steady_clock::now();
      for (auto& input: input_info_) {
        auto& input_id = input.first;
        err = SetInput(*graph_executor, idx++, input_id, jobs);
      }
      auto end = std::chrono::steady_clock::now();
      set_input_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      // DLOG(INFO) << "Inference SetInput: " << std::chrono::duration<double, std::milli>(end - begin).count();
    }

    double infer_ms;
    bool pipeline_exec = false;
    {
      auto begin = std::chrono::steady_clock::now();
      if (Config::pipeline_load && status_[rank] == Status::kWithoutParam) {
        graph_executor->PipeLineLoad();
        graph_executor->PipelineRun();
        pipeline_exec = true;
        
        status_[rank] = Status::kReady;
        model_stat_[static_cast<size_t>(Status::kWithoutParam)].fetch_sub(1, std::memory_order_relaxed);
        model_stat_[static_cast<size_t>(Status::kReady)].fetch_add(1, std::memory_order_relaxed);
      } else {
        graph_executor->Run();
      }
      auto end = std::chrono::steady_clock::now();
      infer_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      // DLOG(INFO) << "Inference run: " << std::chrono::duration<double, std::milli>(end - begin).count();
    }

    double get_output_ms;
    {
      size_t idx = 0;
      auto begin = std::chrono::steady_clock::now();
      for (auto& output : output_info_) {
        for (auto& job : jobs)
          job->GetInferData()->AddOuput();
        auto& output_id = output.first;
        err = GetOutput(*graph_executor, idx++, output_id, jobs);
      }
      auto end = std::chrono::steady_clock::now();
      get_output_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      // DLOG(INFO) << "Inference GetOutput: " << std::chrono::duration<double, std::milli>(end - begin).count();
    }
    auto infer_end = std::chrono::steady_clock::now();

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "[Model Inference]: " << name_ << " (rank " << rank << ") "
       << "set_input_ms=" << set_input_ms << " "
       << (pipeline_exec ? "pipeline_exec infer_ms= " : "infer_ms=") << infer_ms << " "
       << "get_output_ms=" << get_output_ms;
    // if (wait_train_stop_ms != -1) {
    //   ss << " wait_train_stop_ms=" << wait_train_stop_ms;
    // }
    ss << " total_infer_ms=" << std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
    LOG(INFO) << ss.str();

    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferSetInput, set_input_ms);
    if (pipeline_exec) {
      Profiler::Get()->RecordPerf(Profiler::PerfItem::InferPipelineExec, infer_ms);
    } else {
      Profiler::Get()->RecordPerf(Profiler::PerfItem::InferExec, infer_ms);
    }
    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferGetOutput, get_output_ms);
    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferRealBatchSize, jobs.size());
    for (auto& job : jobs) {
      job->RecordFinished();
      job->RecordProfile();
      auto data = job->GetInferData();
      data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    }
    InferModelStore::InferingDec(executors_[rank].get());
    Controller::Get()->InferResponseInc(jobs.size());
  }}

  std::stringstream exit_log_ss;
  exit_log_ss << "[Model Inference] model " << tvm_graph_->GetModelRank() << " worker " << rank << " exit";
  // Profiler::Get()->RecordEvent(Profiler::EventItem::InferExit);
  // GraphCache::Get()->DeInitGraphExecutor(name_, graph_executor);
  // if (Config::IsColocateMode()) {
  //   if (TrainLauncher::Get()->GetTrainPid() != -1) {
  //     Controller::Get()->EnterInferChangeMemory(rank);
  //     // double free_memory_mb = graph_executor->GetFreeMemoryMB();
  //     Controller::Get()->InferExit(
  //       graph_executor->GetFreeMemoryMB() / 145);
  //     exit_log_ss << ", waited train " << waited_trains_[rank] << ", current train pid " << TrainLauncher::Get()->GetTrainPid();
  //     Controller::Get()->ExitInferChangeMemory(rank);
  //   }
  // } else if (Config::IsSwitchMode()) {
  //   InferModelStore::Get()->TaskSwitchExit();
  //   // InferModelStore::Get()->task_switch_cv.notify_all();
  //   // InferModelStore::Get()->task_switch_exit_cv.notify_one();
  //   pthread_barrier_wait(&InferModelStore::Get()->task_switch_barrier); // do cancel exit
  // }
  // *worker_running_[rank] = false;
  // waited_trains_[rank] = static_cast<pid_t>(-1);
  // warmup_ = false;

  LOG(INFO) << exit_log_ss.str();
  return true;
}

bool Model::SetInput(tvm::Executor &graph_executor, 
                     size_t idx, const std::string &input_id, 
                     const std::vector<std::shared_ptr<Job>> &jobs) {
  // check input shape
  auto input_dtype = jobs[0]->GetInferData()->GetInputDType(idx);
  auto batch_shape = jobs[0]->GetInferData()->GetInputShape(idx);
  batch_shape[0] = jobs.size();

  CHECK_EQ(batch_shape.size(), input_info_[input_id].first.size()) << "input shape dimension mismatch";
  CHECK_LE(batch_shape[0], input_info_[input_id].first[0]) << "out of model batch size";
  for (size_t i = 1; i < batch_shape.size(); i++) {
    CHECK_EQ(batch_shape[i], input_info_[input_id].first[i])
        << "input shape[" << i << "] mismatch";
  }

  // tvm::runtime::NDArray input_arr = get_input_(input_id, device_);
  auto input_dev = graph_executor.GetInput(input_id);
  CHECK(sta::DLDataTypeEqual(input_dev->dtype, input_dtype)) 
      << "input dtype mismatch " << (int)input_dev->dtype.bits 
      << " vs " << (int)input_dtype.bits;

  auto copy_batch_input_fn = [&jobs](size_t idx, void* data) {
    size_t offset = 0;
    for (auto job : jobs) {
      auto bytes= job->GetInferData()->GetInputBytes(idx);
      std::memcpy(static_cast<char*>(data) + offset,
                  job->GetInferData()->GetInputData(idx),
                  bytes);
      offset += bytes;
    }
  };

  if (device_.device_type == kDLCUDA) {
    auto input_host_buf = graph_executor.GetInputHostBuf(input_id);
    copy_batch_input_fn(idx, input_host_buf->data);
    // input_arr.CopyFrom(input_cpu);
    ::tvm::runtime::NDArray::CopyFromTo(
        input_host_buf, const_cast<DLTensor*>(input_dev), graph_executor.GetExecStream());
    ::tvm::runtime::DeviceAPI::Get(device_)->StreamSync(device_, graph_executor.GetExecStream());
  } else {
    LOG(FATAL) << "unsupport device type " << device_.device_type;
  }
  return true;
}

bool Model::GetOutput(tvm::Executor &graph_executor, 
                      size_t idx, const std::string &output_id, 
                      const std::vector<std::shared_ptr<Job>> &jobs) {
  const DLTensor* output_host_buf;
  if (device_.device_type == kDLCUDA) {
    auto output_dev = graph_executor.GetOutput(output_id);
    output_host_buf = graph_executor.GetOutputHostBuf(output_id);
    ::tvm::runtime::NDArray::CopyFromTo(
        output_dev, const_cast<DLTensor*>(output_host_buf), graph_executor.GetExecStream());
    ::tvm::runtime::DeviceAPI::Get(device_)->StreamSync(device_, graph_executor.GetExecStream());
  } else {
    LOG(FATAL) << "unsupport device type " << device_.device_type;
  }

  CHECK_LE(jobs.size(), static_cast<size_t>(output_host_buf->shape[0])) << "out of model batch size";
  std::vector<int64_t> shape{output_host_buf->shape, output_host_buf->shape + output_host_buf->ndim};
  shape[0] = 1;

  size_t offset = 0;
  size_t output_nbytes = ::tvm::runtime::GetDataSize(*output_host_buf) / output_host_buf->shape[0];
  for (auto& job : jobs) {
    auto data = job->GetInferData();
    data->SetOutputShape(idx, shape);
    data->SetOutputDType(idx, ::tvm::runtime::DLDataType2String(output_host_buf->dtype));
    // copy output to infer data
    auto output = data->MutableOutputData(idx);
    output->resize(output_nbytes);
    std::memcpy(output->data(), 
                static_cast<char*>(output_host_buf->data) + offset, output_nbytes);
    offset += output_nbytes;
  }
  return true;
}




}