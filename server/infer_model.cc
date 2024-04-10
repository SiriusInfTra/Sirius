#include <pthread.h>
#include <server/infer_model_store.h>
#include <server/infer_model.h>
#include <server/controller.h>
#include <server/config.h>
#include <server/resource_manager.h>
#include <server/train_launcher.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <vector>
#include "common/util.h"


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

  auto rmod = 
      ::tvm::runtime::Module::LoadFromFile((model_path / "mod.so").c_str(), "so");

  if (!params.has_value()) {
    tvm_graph_ = std::make_unique<tvm::TVMGraph>(
      InferModelStore::GetModelRank(),
      this,
      name,
      model_path,
      (model_path / "mod.json").c_str(),
      (model_path / "mod.group").c_str(),
      rmod,
      (model_path / "mod.params").c_str(),
      std::vector{device}
    );
  } else {
    tvm_graph_ = std::make_unique<tvm::TVMGraph>(
      InferModelStore::GetModelRank(),
      this,
      name,
      model_path,
      (model_path / "mod.json").c_str(),
      (model_path / "mod.group").c_str(),
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
  CHECK(max_num_worker_ == 1 && num_worker == 1) << "currently, only support one worker";
  CHECK_LT(max_num_worker_, MAX_NUM_WORKER) << "max num worker exceed limit";
  for (size_t i = 0; i < num_worker; i++) {
    auto executor = tvm_graph_->CreateGraphExecutor(i, std::vector{device});
    // InferModelCache::ReserveCache(name);
    // executor->Init(true);
    executors_.push_back(std::move(executor));
    status_.push_back(Status::kWithoutMemory);
    model_stat_[static_cast<size_t>(Status::kWithoutMemory)].fetch_add(1, std::memory_order_relaxed);
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
    pthread_barrier_destroy(&barrier);
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
  infer_count_.fetch_add(1, std::memory_order_relaxed);
  return job_queue_.Put(std::make_shared<InferJob>(data));
}

bool Model::ReclaimMemory(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, std::unique_lock<std::mutex> &model_lock) {
  if (name_ == "dummy") return false;
  CHECK_LT(rank, muts_.size());
  CHECK_LT(rank, executors_.size()) << name_;
  CHECK_LT(rank, status_.size());
  if (status_[rank] == Status::kWithoutMemory) {
    return false; 
  }
  auto &executor = executors_[rank];
  auto &&[cached_groups_id, evict_group_list, succ] = ColdModelCache::Get()
    .PushCacheItem(name_, rank, executor->GetGroupsNbytes(), executor->GetStorageSizeAlign(), cold_cache_lock);
  CHECK(succ);
  for (auto &&[name, evict_groups_id] : evict_group_list) {
    InferModelStore::Get()->GetModel(name)->ClearColdCache(evict_groups_id, rank, cold_cache_lock);
  }
  executor->DeInit(cached_groups_id);
  ChangeStatus(rank, Status::kWithoutMemory);
  return true;
}

void Model::ClearColdCache(const std::vector<size_t> &cold_cached_group_id, int rank, std::unique_lock<std::mutex> &cold_cache_lock) {
  std::unique_lock other_model_lock{muts_[rank]};
  executors_[rank]->ClearColdCached(cold_cached_group_id);
}

void Model::MaybeAdjustTrainAndCache(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, std::unique_lock<std::mutex> &model_lock) {
  CHECK(status_[rank] == Status::kWithoutMemory);
  PROFILE_START(InferWaitBeforeEnterAlloc);
  ResourceManager::InferMemoryChangingLock();
  PROFILE_END(InferWaitBeforeEnterAlloc);
  double free_memory_MB = ResourceManager::GetFreeMemoryMB();
  double cold_cache_free_memory_MB = ColdModelCache::Get().GetColdCacheFreeMemoryMB(free_memory_MB, cold_cache_lock);
  double total_storage_MB = sta::ByteToMB(executors_[rank]->GetMissingStorageSizeAlign());
  if (total_storage_MB > cold_cache_free_memory_MB && !Controller::Get()->IsTrainIdle()) {
    auto wait_train_pid = TrainLauncher::Get()->GetTrainPid();
    // size_t adjust_batch_buffer_nbytes = std::min(
    //   Config::cold_cache_max_capability_nbytes - Config::cold_cache_min_capability_nbytes,
    //   Config::cold_cache_max_capability_nbytes - ColdModelCache::Get().GetCachedNbytes(cold_cache_lock) 
    // );
    double adjust_reserve_mb = ColdModelCache::Get().GetAdjustReserveMemoryMB(cold_cache_lock);
    double adjust_batch_buffer_mb = total_storage_MB - std::max(0.0, cold_cache_free_memory_MB)
                                    + adjust_reserve_mb;
    if (adjust_batch_buffer_mb > 0) {
      PROFILE_START(TrainAdjust);
      bool is_first_adjust = !Controller::Get()->HasFlyingColocateAdjust();
      auto adjust_batch_size = TrainLauncher::Get()->GetAdjustBatchSize(adjust_batch_buffer_mb);
      CHECK_GE(adjust_batch_size, 0);
      int cmd_id = Controller::Get()->ColocateAdjust(0, adjust_batch_size);
      Controller::Get()->WaitColocateAdjustDone(cmd_id);
      PROFILE_END(TrainAdjust);
      if (is_first_adjust) {
        Profiler::Get()->RecordPerf(Profiler::PerfItem::TrainFirstAdjust, PROFILE_DURATRION(TrainAdjust));
      }
      LOG(INFO) << "[Model, Cold Cache Adjust] AllocStorageMaybeAdjust: model " << rank
                << " wait adjust " << PROFILE_DURATRION(TrainAdjust)
                << " wait train pid " << wait_train_pid
                << " | before adjust: free memory " << free_memory_MB
                << " cold cache free memory " << cold_cache_free_memory_MB
                << " reserve memory " << adjust_reserve_mb
                << " adjust_batch_buffer_mb " << adjust_batch_buffer_mb
                << " delta batch size " << adjust_batch_size << ".";
    } else {
      LOG(INFO) << "[Model, Cold Cache Adjust] AllocStorageMaybeAdjust: model " << rank << " , skip adjust";
    }
    free_memory_MB = ResourceManager::GetFreeMemoryMB();
  }
  LOG(INFO) << "[Model, Cold Cache Adjust] after adjust, "
            << "free memory " << free_memory_MB
            << " cold cache free memory " << ColdModelCache::Get().GetColdCacheFreeMemoryMB(free_memory_MB, cold_cache_lock)
            << " model required " << total_storage_MB;
  if (total_storage_MB > free_memory_MB) {
    long capacity = static_cast<long>(ColdModelCache::Get().GetCachedNbytes(cold_cache_lock)) - static_cast<long>((total_storage_MB - free_memory_MB) * 1024 * 1024);
    auto evict_models = ColdModelCache::Get().GetEvictModels(capacity, name_, cold_cache_lock);
    for (auto &&[name, cached_groups_id] : evict_models) {
      InferModelStore::Get()->GetModel(name)->ClearColdCache(cached_groups_id, rank, cold_cache_lock);
    }
    LOG(INFO) << "[Model, Cold Cache Adjust] after adjust, furthur evict model to make room for model "
              << name_ << " rank " << rank
              << " current cache nbytes " 
              << sta::ByteDisplay(ColdModelCache::Get().GetCachedNbytes(cold_cache_lock));
  }
  ResourceManager::InferMemoryChangingUnlock();
}

bool Model::SetupMemory(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, std::unique_lock<std::mutex> &model_lock) {
  CHECK(status_[rank] == Status::kWithoutMemory);
  ColdModelCache::Get().PopCacheItem(name_, rank, cold_cache_lock);
  if (Config::IsColocateMode() && Config::ondemand_adjust) {
    PROFILE_START(InferAdjustAlloc);
    if (!Config::colocate_config.skip_malloc) MaybeAdjustTrainAndCache(rank, cold_cache_lock, model_lock);
    PROFILE_END(InferAdjustAlloc);
  }
  executors_[rank]->Init(false);
  ChangeStatus(rank, Status::kWithoutParam);
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

  if (barrier != nullptr) {
    int err = pthread_barrier_wait(barrier);
    CHECK(err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD) << "err: " << err << ".";
  } 

  while (!InferModelStore::Initialized()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  LOG(INFO) << "[Model Inference] " << name_ << " (rank " << rank << ") start inference";
  {
    auto reserved_lock = WarmModelCache::ReserveCache(name_, rank);
    CHECK(status_[rank] == Status::kWithoutMemory);
    executors_[rank]->Init(true);
    ChangeStatus(rank, Status::kReady);
  }
  LOG(INFO) << "[Model Inference] " << name_ << " (rank " << rank << ") Init Success";

  // bool first_exec = true;
  
  // auto last_get_batch_time = std::chrono::steady_clock::now();
  auto last_infer_time = Profiler::Now();
  while (true) {             
    auto jobs = job_queue_.GetBatch(batch_size_, 10, 10);
    if (jobs.empty()) {
      auto idle_mill = Profiler::MilliFrom(last_infer_time);
      infer_idle_mills_[rank].store(idle_mill, std::memory_order_relaxed);
      continue;
    }


    // let cache serve models in a fifo manner
    auto reserve_cache_begin = Profiler::Now();
    auto reserved_lock = WarmModelCache::OrderedReserveCache(name_, rank, jobs);
    auto reserve_cache_ms = Profiler::MilliFrom(reserve_cache_begin);

    // [switch mode] before infering, first claim infering execution
    InferModelStore::InferingInc(executors_[rank].get());

    // lock to avoid be interrupted by memory reclaim
    auto cold_cache_lock = ColdModelCache::Get().Lock();
    LOG(INFO) << "[Model Inference] " << name_ << " (rank " << rank << ") Acquire cold cache lock success.";
    std::unique_lock lock{muts_[rank]};
    last_infer_time = Profiler::Now();
    infer_idle_mills_[rank].store(0, std::memory_order_relaxed);
    InferModelStore::UpdateLastInferTime();

    double setup_mem_ms = 0;
    bool setup_memory = false;
    {
      if (status_[rank] == Status::kWithoutMemory) {

        auto begin = Profiler::Now();
        SetupMemory(rank, cold_cache_lock, lock);
        setup_mem_ms = Profiler::MilliFrom(begin);
        setup_memory = true;
      }
    }
    cold_cache_lock.unlock();

    // last_infer_times_[rank] = Profiler::Now();
    DLOG(INFO) << "[Model Inference] GetBatch " << jobs.size() << "/" << batch_size_;

    bool err = false;
    auto infer_begin = std::chrono::steady_clock::now();

    double set_input_ms;
    {
      size_t idx = 0;
      PROFILE_START(InferSetInput);
      for (auto& input: input_info_) {
        auto& input_id = input.first;
        err = SetInput(*graph_executor, idx++, input_id, jobs);
      }
      PROFILE_END(InferSetInput);
      set_input_ms = PROFILE_DURATRION(InferSetInput);
    }

    double loading_ms;
    bool load_param = false;
    {
      if (!Config::pipeline_load && status_[rank] == Status::kWithoutParam) {
        PROFILE_START(InferLoadParam);
        graph_executor->LoadParams(false, false);
        PROFILE_END(InferLoadParam);

        load_param = true;
        loading_ms = PROFILE_DURATRION(InferLoadParam);

        ChangeStatus(rank, Status::kReady);
      }
    }

    double infer_ms;
    bool pipeline_exec = false;
    {
      if (Config::pipeline_load && status_[rank] == Status::kWithoutParam) {
        PROFILE_START(InferPipelineExec);
        graph_executor->PipeLineLoad();
        graph_executor->PipelineRun();
        pipeline_exec = true;
        PROFILE_END(InferPipelineExec);
        infer_ms = PROFILE_DURATRION(InferPipelineExec);

        ChangeStatus(rank, Status::kReady);
      } else {
        PROFILE_START(InferExec);
        CHECK(status_[rank] == Status::kReady);
        graph_executor->Run();
        PROFILE_END(InferExec);
        infer_ms = PROFILE_DURATRION(InferExec);
      }
    }

    double get_output_ms;
    {
      size_t idx = 0;
      PROFILE_START(InferGetOutput);
      for (auto& output : output_info_) {
        for (auto& job : jobs)
          job->GetInferData()->AddOuput();
        auto& output_id = output.first;
        err = GetOutput(*graph_executor, idx++, output_id, jobs);
      }
      PROFILE_END(InferGetOutput);
      get_output_ms = PROFILE_DURATRION(InferGetOutput);
    }
    auto infer_end = std::chrono::steady_clock::now();

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "[Model Inference]: " << name_ << " (rank " << rank << ") "
       << "set_input_ms=" << set_input_ms << " "
       << (pipeline_exec ? "pipeline_exec infer_ms= " : "infer_ms=") << infer_ms << " "
       << "get_output_ms=" << get_output_ms;
    if (setup_memory) { ss << " setup_mem_ms=" << setup_mem_ms; }
    if (load_param) { ss << " loading_ms=" << loading_ms; }

    ss << " total_infer_ms=" << std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
    if (WarmModelCache::Enable()) { ss << " | reserve_cache_ms=" << reserve_cache_ms; }
    LOG(INFO) << ss.str();

    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferRealBatchSize, jobs.size());
    for (auto& job : jobs) {
      job->RecordFinished();
      job->RecordProfile();
      auto data = job->GetInferData();
      data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    }
    InferModelStore::InferingDec(executors_[rank].get());
    Controller::Get()->InferResponseInc(jobs.size());
  }

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