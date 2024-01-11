#include "logging_as_glog.h"
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <regex>

#include <sta/dtype_helper.h>

#include "cache.h"
#include "model_infer_store.h"
#include "model_train_store.h"
#include "controller.h"
#include "profiler.h"
#include "config.h"


namespace colserve {

namespace {
std::vector<std::string> ParseModelName(const std::string &model) {
  std::regex r{"([a-zA-Z0-9_]+)(\\[([0-9]+)\\])?"};
  std::smatch match;
  CHECK(std::regex_match(model, match, r)) << "model name " << model << " not match";
  CHECK_EQ(match.size(), 4);
  if (match[3].str().empty()) {
    return {match[1].str()};
  } else {
    CHECK_GE(std::stoi(match[3].str()), 1);
    std::vector<std::string> ret{match[1].str()};
    for (int i = 1; i < std::stoi(match[3].str()); i++) {
      ret.push_back(match[1].str() + "-" + std::to_string(i));
    }
    return ret;
  }
}
}

std::unique_ptr<ModelInferStore> ModelInferStore::model_infer_store_;
std::atomic<size_t> ModelInferStore::model_rank = 0;

ModelInferStore* ModelInferStore::Get() {
  if (model_infer_store_ == nullptr) {
    LOG(FATAL) << "ModelInferStore not initialized";
  }
  return model_infer_store_.get();
}

void ModelInferStore::Init(const std::filesystem::path &infer_store_path) {
  LOG(INFO) << "ModelInferStore start initializing";
  model_infer_store_ = std::make_unique<ModelInferStore>();
  model_infer_store_->models_["dummy"] = std::make_unique<Model>();

  // model_name -> [key -> value]
  std::map<std::string, std::map<std::string, std::string>> models;
  
  std::filesystem::path config_file_path;
  if (Config::infer_model_config_path.at(0) == '/') {
    config_file_path = Config::infer_model_config_path;
  } else {
    config_file_path = infer_store_path / Config::infer_model_config_path;
  }

  std::ifstream config_file(config_file_path);
  if (config_file.good()) {
    std::string cur_model;
    for (std::string line; std::getline(config_file, line);) {
      if (line.empty()) continue;
      if (line[0] == '#') continue;
      std::istringstream iss(line);
      if (line[0] != ' ' && line[0] != '\n' && line[0] != '\t') {
        iss >> cur_model;
        models[cur_model] = {};
      } else {
        std::string key, value;
        iss >> key >> value;
        models[cur_model][key] = value;
      }
    }
    for (auto &model: models) {
      std::stringstream ss;
      ss << "[" << model.first << "] "; 
      CHECK(model.second.count("path"));
      CHECK(model.second.count("device"));
      CHECK(model.second.count("batch-size"));
      if (!model.second.count("num-worker")) {
        model.second["num-worker"] = "1";
      }
      if (!model.second.count("max-worker")) {
        model.second["max-worker"] = "1";
      }
      for (auto &kv : model.second) {
        ss << kv.first << "=" << kv.second << " ";
      }
      LOG(INFO) << "[ModelInferStore] Read from config file: " << ss.str();
    }
  }

  // mod.so, mod.json, mod.params should be in the model directory
  for (auto &model : models) {
    auto model_path = infer_store_path / model.second["path"];
    CHECK(std::filesystem::exists(model_path)) << "ModelInferStore: " << model_path << " not exist";
    CHECK(std::filesystem::is_directory(model_path)) << model_path << " is not a directory";
    auto model_params = tvm::GraphExecutorFactory::LoadParamsAsTVMArray(
        (model_path / "mod.params").c_str());
    
    for (auto model_name : ParseModelName(model.first)) {
      auto model_device = model.second["device"];
      size_t batch_size = std::stoi(model.second["batch-size"]);
      DLDevice device;
      if (model_device == "cuda") {
        device = DLDevice{kDLCUDA, 0};
      } else {
        LOG(FATAL) << model_name << " unsupport device type " << model_device;
      }
      model_infer_store_->models_[model_name] = 
          std::make_unique<Model>(model_name, model_path, model_params,
                                  device, batch_size, 
                                  std::stoi(model.second["num-worker"]),
                                  std::stoi(model.second["max-worker"]));
    }
    LOG(INFO) << "ModelInferStore: "<< "Add " << model.first << ":" << model.second["device"]
              << ", batch-size=" << model.second["batch-size"];
    model_rank++;
  }

  if (Config::IsSwitchMode()) {
    model_infer_store_->task_switch_control_.reset(new std::thread([&]() {
      while (true) {
        if (Controller::Get()->IsInferIdle()) {
          // first ensure all not more infer workers
          std::unique_lock lock{model_infer_store_->task_switch_mutex_};
          ModelInferStore::Get()->task_switch_cv.wait(lock, [&]() {
            return model_infer_store_->task_switch_enter_cnt_ > 0 
                && (model_infer_store_->task_switch_control_cnter_ == 
                    static_cast<int>(ModelInferStore::TaskSwitchStatus::kNotAddWorker));
          });

          pthread_barrier_init(&model_infer_store_->task_switch_barrier, nullptr, 
              model_infer_store_->task_switch_enter_cnt_ + 1);
          // enter task switch prepare exit stage
          model_infer_store_->task_switch_control_cnter_ = static_cast<int>(ModelInferStore::TaskSwitchStatus::kPrepareExit);
          // LOG(INFO) << "[ModelInferStore]: try task switch " << model_infer_store_->task_switch_enter_cnt_;
          auto t0 = Profiler::Get()->Now();

          // let all infer enter task switch prepare exit stage
          pthread_barrier_wait(&model_infer_store_->task_switch_barrier);
          auto wait_task_exit_ms = Profiler::Get()->MilliFrom(t0);
          LOG(INFO) << "[ModelInferStore] [Task Switch]: wait for inference threads " << wait_task_exit_ms << " ms "
                    << " wait up to " << Config::task_switch_delay_ms << " ms";
          if (wait_task_exit_ms < Config::task_switch_delay_ms) {
            auto delay_us = static_cast<int>((Config::task_switch_delay_ms - wait_task_exit_ms) * 1000);
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
          }
          if (Controller::Get()->IsInferIdle()) {
            model_infer_store_->task_switch_control_cnter_ = static_cast<int>(ModelInferStore::TaskSwitchStatus::kExit);
            Profiler::Get()->RecordPerf(Profiler::PerfItem::InferNumModelOnSwitch, model_infer_store_->task_switch_enter_cnt_);
            // all infer know result
            pthread_barrier_wait(&model_infer_store_->task_switch_barrier);
            LOG(INFO) << "[ModelInferStore] [Task Switch]: task switch to train | " << Controller::Get()->GetInferStatusStr();
            // all infer do task switch
            pthread_barrier_wait(&model_infer_store_->task_switch_barrier);
            Controller::Get()->ResumeTrain();
          } else {
            model_infer_store_->task_switch_control_cnter_ = static_cast<int>(ModelInferStore::TaskSwitchStatus::kCancelExit);
            // all infer know result
            pthread_barrier_wait(&model_infer_store_->task_switch_barrier);
            // all infer cancel task switch
            pthread_barrier_wait(&model_infer_store_->task_switch_barrier); 
            LOG(INFO) << "[ModelInferStore] [Task Switch]: task switch cancel exit | " << Controller::Get()->GetInferStatusStr();
          }
          model_infer_store_->task_switch_control_cnter_ = static_cast<int>(ModelInferStore::TaskSwitchStatus::kNotAddWorker);
          model_infer_store_->task_switch_cv.notify_all();
          pthread_barrier_destroy(&model_infer_store_->task_switch_barrier);
        } else {
          // Controller::Get()->LogInferStatus();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }));
  }

  LOG(INFO) << "ModelInferStore initialized";
}

Model* ModelInferStore::GetModel(const std::string &name) {
  auto it = models_.find(name);
  if (it == models_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

size_t ModelInferStore::NumJobs() {
  size_t num_jobs = 0;
  for (auto &model : models_) {
    num_jobs += model.second->NumJobs();
  }
  return num_jobs;
}

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
    graph_executor_factory_ = std::make_unique<tvm::GraphExecutorFactory>(
      ModelInferStore::model_rank.fetch_add(1, std::memory_order_relaxed),
      this,
      name,
      (model_path / "mod.json").c_str(),
      rmod,
      (model_path / "mod.params").c_str(),
      std::vector{device}
    );
  } else {
    graph_executor_factory_ = std::make_unique<tvm::GraphExecutorFactory>(
      ModelInferStore::model_rank.fetch_add(1, std::memory_order_relaxed),
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
    // graph_executor_factory_->ResetParamStorage();
    LOG(INFO) << "[Model]: " << name << " task switch by pipelining load/run";
  }

  // config infer scaling
  scale_up_queue_time_ = 200; // no used
  scale_down_idle_time_ = Config::infer_model_max_idle_ms;
  warmup_ = Config::has_warmup;
  // max_num_worker_ = 5;
  infer_workers_.resize(max_num_worker_);
  worker_running_.resize(max_num_worker_);
  waited_trains_.resize(max_num_worker_);
  for (size_t i = 0; i < max_num_worker_; i++) {
    graph_executor_pool_.push_back(graph_executor_factory_->CreateGraphExecutor(i));
    worker_running_[i] = std::make_unique<std::atomic<bool>>(false);
    waited_trains_[i] = static_cast<pid_t>(-1);
  }

  if (Config::colocate_config.skip_malloc || Config::colocate_config.skip_loading) {
    for (size_t i = 0; i < max_num_worker_; i++) {
      graph_executor_pool_[i].get()->FakeInit(Config::colocate_config.skip_malloc, Config::colocate_config.skip_loading);
    }
  }

  num_worker_ = 0;
  for (size_t i = 0; i < num_worker; i++) {
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, 2);
    *worker_running_[i] = true;
    // infer_workers_.push_back(std::make_unique<std::thread>(&Model::Inference, this, &barrier));
    infer_workers_[i].reset(new std::thread{&Model::Inference, this, i, &barrier});
    if (Config::IsSwitchMode()) pthread_barrier_wait(&barrier);
    pthread_barrier_wait(&barrier);
  }

  if (Config::IsColocateMode() || Config::IsSwitchMode()) {
    job_monitor_.reset(new std::thread{&Model::MonitorJob, this});
  }
}

bool Model::AddJob(network::InferHandler::InferData* data) {
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

void Model::InitMetaInfo() {
  auto [shape_info, dtype_info] = 
      graph_executor_factory_->GetInputInfo();
  for (const auto &kv : shape_info) {
    auto shape = shape_info[kv.first];
    auto dtype = dtype_info[kv.first];
    input_info_[kv.first] = std::make_pair(shape, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); i++) 
      ss << shape[i] << " ";
    LOG(INFO) << "Model Input: " << name_ << " " << kv.first << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape[0], batch_size_) << "batch size mismatch";
  }

  std::tie(shape_info, dtype_info) = graph_executor_factory_->GetOutputInfo();
  for (const auto &kv : shape_info) {
    auto shape = shape_info[kv.first];
    auto dtype = dtype_info[kv.first];
    output_info_[kv.first] = std::make_pair(shape, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); i++)
      ss << shape[i] << " ";
    LOG(INFO) << "Model Output: " << name_ << " " << kv.first << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape[0], batch_size_) << "batch size mismatch";
  }
}

bool Model::Inference(uint32_t rank, pthread_barrier_t* barrier) {
  LOG(INFO) << "Model " << name_ << " inference thread start";
  if (Config::IsSwitchMode()) {
    ModelInferStore::Get()->TaskSwitchEnter();
    CHECK(barrier != nullptr);
    pthread_barrier_wait(barrier);
  }
  // auto graph_executor = graph_executor_factory_->CreateGraphExecutor();
  auto graph_executor = graph_executor_pool_[rank].get();
  GraphCache::Get()->InitGraphExecutor(name_, graph_executor);

  if (barrier != nullptr) pthread_barrier_wait(barrier);

  bool first_exec = true;
  num_worker_.fetch_add(1, std::memory_order_relaxed);
  auto last_get_batch_time = std::chrono::steady_clock::now();
  LOG(INFO) << "[Model Inference] " << name_ << " (rank " << rank << ") start inference";
  while (true) {                                                    
    if (Config::IsColocateMode() && Profiler::MilliFrom(last_get_batch_time) >= GetMaxIdleTime()) {
      uint32_t num_worker = num_worker_;
      if (num_worker - 0 > 0) { /* check if num_worker reduce */
        // LOG(INFO) << "num_worker " << num_worker;
        auto ok = num_worker_.compare_exchange_strong(num_worker, num_worker - 1,
            std::memory_order_relaxed);
        if (ok) break;
      }
    } else if (Config::IsSwitchMode() && ModelInferStore::Get()->TaskSwitchControlCnter() == 
        static_cast<int>(ModelInferStore::TaskSwitchStatus::kPrepareExit)) {
      // wait all infer enter task switch prepare exit stage
      pthread_barrier_wait(&ModelInferStore::Get()->task_switch_barrier);
      // wait task switch result
      pthread_barrier_wait(&ModelInferStore::Get()->task_switch_barrier);
      if (ModelInferStore::Get()->TaskSwitchControlCnter() == 
          static_cast<int>(ModelInferStore::TaskSwitchStatus::kCancelExit)) { // not exit
        // do cancel exit        
        pthread_barrier_wait(&ModelInferStore::Get()->task_switch_barrier);
      } else { // exit
        num_worker_.fetch_sub(1, std::memory_order_relaxed);
        break;
      }
    }

    // TODO dynamic batching
    auto jobs = job_queue_.GetBatch(batch_size_, 10, 10);
    if (jobs.empty())
      continue;
    last_get_batch_time = std::chrono::steady_clock::now();
    DLOG(INFO) << "[Model Inference] GetBatch " << jobs.size() << "/" << batch_size_;

    auto infer_begin = std::chrono::steady_clock::now();

    bool err = false;

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
    {
      auto begin = std::chrono::steady_clock::now();
      if (Config::pipeline_load && first_exec) {
        graph_executor->PipelineRun();
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
       << (Config::pipeline_load && first_exec ? "pipeline_exec infer_ms= " : "infer_ms=") << infer_ms << " "
       << "get_output_ms=" << get_output_ms;
    // if (wait_train_stop_ms != -1) {
    //   ss << " wait_train_stop_ms=" << wait_train_stop_ms;
    // }
    ss << " total_infer_ms=" << std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
    LOG(INFO) << ss.str();

    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferSetInput, set_input_ms);
    if (Config::pipeline_load && first_exec) {
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
    Controller::Get()->InferResponseInc(jobs.size());
    first_exec = false;
  }

  std::stringstream exit_log_ss;
  exit_log_ss << "[Model Inference] model " << graph_executor_factory_->GetModelRank() << " worker " << rank << " exit";
  Profiler::Get()->RecordEvent(Profiler::EventItem::InferExit);
  GraphCache::Get()->DeInitGraphExecutor(name_, graph_executor);
  if (Config::IsColocateMode()) {
    Controller::Get()->InferExit(
      (waited_trains_[rank] != -1 && waited_trains_[rank] == ModelTrainStore::Get()->GetTrainPid()) ? graph_executor_pool_.front()->GetAdjustBatchSize() : 0);
    exit_log_ss << ", waited train " << waited_trains_[rank] << ", current train pid " << ModelTrainStore::Get()->GetTrainPid();
  } else if (Config::IsSwitchMode()) {
    ModelInferStore::Get()->TaskSwitchExit();
    // ModelInferStore::Get()->task_switch_cv.notify_all();
    // ModelInferStore::Get()->task_switch_exit_cv.notify_one();
    pthread_barrier_wait(&ModelInferStore::Get()->task_switch_barrier); // do cancel exit
  }
  *worker_running_[rank] = false;
  waited_trains_[rank] = static_cast<pid_t>(-1);
  warmup_ = false;

  LOG(INFO) << exit_log_ss.str();
  return true;
}

bool Model::SetInput(tvm::GraphExecutor &graph_executor, 
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
  CHECK(sta::DLDataTypeEqual(input_dev->dtype, input_dtype)) << "input dtype mismatch " << (int)input_dev->dtype.bits << " vs " << (int)input_dtype.bits;

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

bool Model::GetOutput(tvm::GraphExecutor &graph_executor, 
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

void Model::MonitorJob() {
  while (true) {
    for (size_t i = 0; i < max_num_worker_; i++) {
      if (infer_workers_[i] != nullptr && !*worker_running_[i]) {
        infer_workers_[i]->join();
        infer_workers_[i].reset();
        LOG(INFO) << "[Model Monitor] " << name_ << " decrease worker " << i;
      }
    }

    // auto queue_time = job_queue_.FirstJobQueueTime();
    auto queue_size = job_queue_.NumJobs();
    // if (queue_time >= scale_up_queue_time_) {
    if (queue_size > batch_size_ * num_worker_) {
      for (size_t i = 0; i < max_num_worker_; i++) {
        if (infer_workers_[i] == nullptr) {
          if (Config::IsColocateMode()) {
            auto t0 = std::chrono::steady_clock::now();
            if (!Config::ondemand_adjust) {
              pid_t infer_waited_train;
              if (!Controller::Get()->IsTrainIdle()) {
                infer_waited_train = ModelTrainStore::Get()->GetTrainPid();
                auto id = Controller::Get()->ColocateAdjust(graph_executor_pool_.front()->GetAdjustBatchSize());
                if (!Config::colocate_config.skip_malloc) {
                  Controller::Get()->WaitColocateAdjustDone(id);
                }
                auto t1 = std::chrono::steady_clock::now();
                LOG(INFO) << "[Model Monitor]: Wait adjust train batch size before add infer "
                          << std::chrono::duration<double, std::milli>(t1-t0).count() << " ms";
                // PROFILE_END(TrainAdjust, 1);
                Profiler::Get()->RecordEvent(Profiler::EventItem::TrainAdjustStart, t0);
                Profiler::Get()->RecordEvent(Profiler::EventItem::TrainAdjustEnd, t1);
                Profiler::Get()->RecordPerf(Profiler::PerfItem::TrainAdjust, t0, t1);
              } else {
                infer_waited_train = -1;
                LOG(INFO) << "[Model Monitor]: add infer skip wait, train is idle";
              }
              waited_trains_[i] = infer_waited_train;
            }
          } else { // switch mode
            std::unique_lock lock{ModelInferStore::Get()->TaskSwitchMutex()};
            ModelInferStore::Get()->task_switch_cv.wait(lock, [&]() {
              return ModelInferStore::Get()->TaskSwitchControlCnter() >= 
                  static_cast<int>(ModelInferStore::TaskSwitchStatus::kNotAddWorker);
            });
            ModelInferStore::Get()->MutableTaskSwitchControlCnter().fetch_add(1, std::memory_order_relaxed);
            lock.unlock();
            auto begin = std::chrono::steady_clock::now();
            Controller::Get()->WaitTrainNotRunning();
            auto wait_train_stop_ms = Profiler::MilliFrom(begin);
            // ModelInferStore::Get()->MutableTaskSwitchControlCnter().fetch_sub(1, std::memory_order_relaxed);
            LOG(INFO) << "[Model Monitor]: " << name_ << " wait for train stop " << wait_train_stop_ms << " ms";
          }
          pthread_barrier_t barrier;
          pthread_barrier_init(&barrier, nullptr, 2);
          *worker_running_[i] = true;
          infer_workers_[i].reset(new std::thread{&Model::Inference, this, i, &barrier});
          if (Config::IsSwitchMode()) {
            pthread_barrier_wait(&barrier);
            ModelInferStore::Get()->MutableTaskSwitchControlCnter().fetch_sub(1, std::memory_order_relaxed);
            ModelInferStore::Get()->task_switch_cv.notify_all();
          }
          pthread_barrier_wait(&barrier);
          Profiler::Get()->RecordEvent(Profiler::EventItem::AddInfer);
          LOG(INFO) << "[Model Monitor] " << name_ << " increase worker " << i;
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
          break;
        }
      }
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
}

} // namespace colserve