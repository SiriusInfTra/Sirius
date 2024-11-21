#include <server/logging_as_glog.h>
#include <server/model_store/infer_model_store.h>
#include <server/model_store/infer_model.h>
#include <server/model_store/model_cache.h>
#include <server/control/controller.h>
#include <server/config.h>
#include <server/resource_manager.h>
#include <server/train_launcher.h>
#include <server/train_adjuster.h>

#include <common/util.h>
#include <common/sm_partition.h>
#include <common/device_manager.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <vector>
#include <regex>
#include <unordered_map>
#include <cmath>
#include <pthread.h>

constexpr bool SKIP_MULTI_OUTPUT = false;

namespace colserve {

std::string GetModelNameWithoutDuplicatedId(const std::string &model_name) {
  std::regex r{"([a-zA-Z0-9_]+)(-[0-9]+)?"};
  std::smatch match;
  CHECK(std::regex_match(model_name, match, r)) 
      << "model name " << model_name << " is not valid";
  CHECK_EQ(match.size(), 3);
  CHECK(!match[1].str().empty());
  return match[1].str();
}

std::array<std::atomic<int>, static_cast<size_t>(Model::Status::kNumStatus)> 
    Model::model_stat_{0, 0, 0};

std::mutex Model::estimate_tpc_mut_;
std::condition_variable Model::estimate_tpc_cv_;
bool Model::estimating_tpc_ = false;

int Model::GetPreEstimatedTPC(const std::string &model_name) {
  auto model_name_without_dup_id = GetModelNameWithoutDuplicatedId(model_name);
  std::unordered_map<std::string, int> tpc_map = {
    {"resnet152", 27},
    {"densenet161", 13},
    {"efficientvit_b2", 31},
    {"efficientnet_v2_s", 31},
    {"distilgpt2", 31},
    {"distilbert_base", 33},
  };
  CHECK(tpc_map.count(model_name_without_dup_id) > 0) 
      << "model " << model_name << " not found";
  return tpc_map[model_name_without_dup_id];
}

Model::Model(const std::string &name, 
             const std::filesystem::path &model_path,
             std::optional<const std::map<std::string, tvm::TVMArray>> params,
             DLDevice device, size_t batch_size, 
             size_t num_worker, size_t max_num_worker)
    : name_(name), device_(device), 
      batch_size_(batch_size), 
      max_num_worker_(max_num_worker) {
  // DLOG(INFO) << "Model " << name << " start initializing from " << model_path;
  CHECK(std::filesystem::exists(model_path / "mod.so"));
  CHECK(std::filesystem::exists(model_path / "mod.json"));
  CHECK(std::filesystem::exists(model_path / "mod.params"));

  auto rmod = ::tvm::runtime::Module::LoadFromFile(
      (model_path / "mod.so").c_str(), "so");

  if (!params.has_value()) {
    tvm_graph_ = std::make_unique<tvm::TVMGraph>(
      InferModelStore::GetModelRank(),
      this,
      name,
      model_path,
      (model_path / "mod.json").c_str(),
      (model_path / "mod.group").c_str(),
      rmod,
      (model_path / "mod.params").c_str()
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
      params.value()
    );
  }

  InitMetaInfo();

  if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    // tvm_graph_->ResetParamStorage();
    LOG(INFO) << "[Model]: " << name << " task switch by pipelining load/run";
  }

  CHECK_LE(num_worker, max_num_worker);
  CHECK(max_num_worker_ == 1 && num_worker == 1) 
      << "currently, only support one worker";
  CHECK_LT(max_num_worker_, MAX_NUM_WORKER) 
      << "max num worker exceed limit";

  for (size_t i = 0; i < num_worker; i++) {
    auto executor = tvm_graph_->CreateGraphExecutor(i, std::vector{device});
    // InferModelCache::ReserveCache(name);
    // executor->Init(true);
    executors_.push_back(std::move(executor));
    status_.push_back(Status::kWithoutMemory);
    model_stat_[static_cast<size_t>(Status::kWithoutMemory)].fetch_add(
        1, std::memory_order_relaxed);
  }
  infer_workers_.resize(max_num_worker_);

  LOG_IF(INFO, Config::log_infer_model_init)
      << "[Model Init] " << name << " initilized " << num_worker << " executor";

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

  if (!Config::estimate_infer_model_tpc) {
    required_num_tpc_ = Model::GetPreEstimatedTPC(name);
  }
}

void Model::InitMetaInfo() {
  auto [shape_info, dtype_info] = tvm_graph_->GetInputInfo();
  for (const auto &kv : shape_info) {
    auto shape = shape_info[kv.first];
    auto dtype = dtype_info[kv.first];
    input_info_[kv.first] = std::make_pair(shape, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); i++) 
      ss << shape[i] << " ";
    LOG_IF(INFO, Config::log_infer_model_init) 
        << "[Model Init] Input: " << name_ << " " << kv.first 
        << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape[0], batch_size_) << "batch size mismatch";
  }

  std::tie(shape_info, dtype_info) = tvm_graph_->GetOutputInfo();
  for (const auto &kv : shape_info) {
    auto shape = shape_info[kv.first];
    auto dtype = dtype_info[kv.first];
    std::stringstream ss;
    for (size_t i = 0; i < shape.size(); i++)
      ss << shape[i] << " ";
    LOG_IF(INFO, Config::log_infer_model_init) 
        << "[Model Init] Output: " << name_ << " " << kv.first 
        << " shape [ " << ss.str()  << "] dtype " << dtype;
    
    CHECK_EQ(shape[0], batch_size_)
        << "[Model Init] Output: " << name_ << " " << kv.first  
        << ", batch size mismatch";
    if (SKIP_MULTI_OUTPUT && !output_info_.empty()) {
      LOG(INFO) << "[Model Init] Output: " << name_ << " " << kv.first 
                << ", multi output, skiped in output_info_";
    } else {
      output_info_[kv.first] = std::make_pair(shape, dtype);
    }
  }
}

bool Model::AddJob(network::InferHandler::InferData* data) {
  // LOG(INFO) << "model " << name_ << " add job";
  infer_count_.fetch_add(1, std::memory_order_relaxed);
  return job_queue_.Put(std::make_shared<InferJob>(data));
}

bool Model::ReclaimMemory(size_t rank, Model *source_model) {
  if (name_ == "dummy") return false;
  
  auto cold_cache_lock = ColdModelCache::Get(device_.device_id)->Lock();
  auto model_lock = std::unique_lock{muts_[rank]};
  return ReclaimMemory(rank, cold_cache_lock, model_lock, source_model);
}

bool Model::ReclaimMemory(size_t rank, 
                          std::unique_lock<std::mutex> &cold_cache_lock, 
                          std::unique_lock<std::mutex> &model_lock, 
                          Model *source_model) {
  if (name_ == "dummy") return false;
  CHECK_LT(rank, muts_.size());
  CHECK_LT(rank, executors_.size()) << name_;
  CHECK_LT(rank, status_.size());
  if (status_[rank] == Status::kWithoutMemory) {
    return false; 
  }

  auto &executor = executors_[rank];
  auto cold_model_cache = ColdModelCache::Get(device_.device_id);
  auto &&[cached_groups_id, evict_group_list, succ] = 
    cold_model_cache->PushCacheItem(name_, rank, 
                                    tvm_graph_->GetGroupsNbytes(), 
                                    tvm_graph_->GetStorageAlignedNbytes(), 
                                    cold_cache_lock, source_model);
  CHECK(succ);
  for (auto &&[name, evict_groups_id] : evict_group_list) {
    InferModelStore::Get()->GetModel(name)->
      ClearColdCache(evict_groups_id, rank, cold_cache_lock);
  }
  executor->DeInit(cached_groups_id);
  ChangeStatus(rank, Status::kWithoutMemory);

  return true;
}

void Model::ClearColdCache(const std::vector<size_t> &cold_cached_group_id, 
                           int rank, 
                           std::unique_lock<std::mutex> &cold_cache_lock) {
  auto t0 = std::chrono::steady_clock::now();
  auto cache_before = ResourceManager::GetFreeMemoryMB(
      sta::DeviceManager::GetCurrentDevice(), false);
  std::unique_lock other_model_lock{muts_[rank]};
  executors_[rank]->ClearColdCached(cold_cached_group_id);
  auto dur = std::chrono::steady_clock::now() - t0;
  auto cache_after = ResourceManager::GetFreeMemoryMB(
      sta::DeviceManager::GetCurrentDevice(), false);

  LOG(INFO) << "[ClearCache] cost " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() 
    << "ms: " << cache_before << "MB -> " << cache_after << "MB"
    << ", cached " 
    << sta::PrintByte(ColdModelCache::Get(device_.device_id)
        ->GetCachedNbytes(cold_cache_lock))
    << ", cap " 
    << sta::PrintByte(ColdModelCache::Get(device_.device_id)
        ->GetCacheSizeMBUnsafe()) 
    << ".";
}

bool Model::MaybeAdjustTrainAndCache(size_t rank, 
                                     std::unique_lock<std::mutex> &cold_cache_lock, 
                                     std::unique_lock<std::mutex> &model_lock) {
  CHECK(status_[rank] == Status::kWithoutMemory);
  if (ctrl::Controller::Get()->IsTrainIdle()) {
    LOG(INFO) << "[Model, Cold Cache Adjust] "
              << "AllocStorageMaybeAdjust: model " << rank
              << " train idle, skip adjust";
    return false;
  }
#if ADJUST_WITH_FLYING
  // batching adjust with flying adjusts, deprecated currently

  // if (try_lock_memory_changing_succ = ResourceManager::InferChangeMemoryTryLock()) {
  //   if (Controller::Get()->HasFlyingColocateAdjust()) {
  //     double total_storage_MB = sta::ByteToMB(executors_[rank]->GetMissingStorageSizeAlign());
  //     double free_memory_MB = ResourceManager::GetFreeMemoryMB(true);
  //     double cold_cache_free_memory_MB = 
  //         ColdModelCache::Get().GetColdCacheFreeMemoryMB(free_memory_MB, cold_cache_lock);
  //     if (total_storage_MB > cold_cache_free_memory_MB && !Controller::Get()->IsTrainIdle()) {
  //       PROFILE_START(TrainAdjust);
  //       auto adjust_batch_size = TrainLauncher::Get()->
  //         GetAdjustBatchSize(total_storage_MB);
  //       auto cmd_id = Controller::Get()->ColocateAdjust(rank, adjust_batch_size);
  //       // LOG(INFO) << "try adjust, rank " << name_ << " cmd_id " << cmd_id;
  //       Controller::Get()->WaitColocateAdjustDone(cmd_id);
  //       PROFILE_END(TrainAdjust);
  //       LOG(INFO) << "[Model, Cold Cache Adjust] adjust with flyings,"
  //                 << " adjust memory mb " << total_storage_MB
  //                 << " wait adjust " << PROFILE_DURATRION(TrainAdjust);
  //     }
  //   }
  // }
#endif

  auto cold_model_cache = ColdModelCache::Get(device_.device_id);
  cold_model_cache->BlockProfilter();
  size_t total_storage_nbytes = executors_[rank]->GetMissingStorageSizeAlign();
  LOG(INFO) << "[Model, Cold Cache Adjust] "
            << "AllocStorageMaybeAdjust: model " << rank
            << " total_storage_nbytes " << total_storage_nbytes;

  if (!cold_model_cache->TakeSpace(total_storage_nbytes)) {
    /* if: need adjust train / infer cache */
    if (cold_model_cache->GetCacheCapacity(cold_cache_lock)
          < (Config::cold_cache_min_capability_nbytes + total_storage_nbytes)
    ) {
      memory_byte_t new_capacity = 
        (Config::cold_cache_max_capability_nbytes 
         + Config::cold_cache_min_capability_nbytes) / 2;
      // reset memory predict accumulation error
      memory_mb_t require_mb = 
          + (sta::ByteToMB(new_capacity)
          - sta::ByteToMB(cold_model_cache->GetCachedNbytes(cold_cache_lock)))
          - (ResourceManager::GetFreeMemoryMB(sta::DeviceManager::GetCurrentDevice(), true));
      LOG(INFO) << "[Model, Cold Cache Adjust] "
                << "Adjust train " << rank
                << " require " << require_mb << "MB";
      auto adjust_plan = TrainAdjuster::GetInferRequireMemAdjustPlanWithInLock(
          device_.device_id, require_mb, 
          nan(""), cold_cache_lock);
      if (!adjust_plan.empty() || require_mb <= 0.0) {
        if (!adjust_plan.empty()) {
          PROFILE_START(TrainAdjust);
          auto cmd_id = ctrl::Controller::Get()->ColocateInferRequireAdjust(
              rank, device_.device_id, adjust_plan);
          ctrl::Controller::Get()->WaitColocateAdjustDone(cmd_id);
          PROFILE_END(TrainAdjust);
          LOG_IF(INFO, Config::log_memory_adjust) 
              << "[Model, Cold Cache Adjust] "
              << "AllocStorageMaybeAdjust: model " << name_ 
              << " new capacity " << sta::ByteToMB(new_capacity)
              << " wait adjust " << PROFILE_DURATRION(TrainAdjust);
        } else {
          LOG(INFO) << "[MaybeAdjustTrainAndCache]  require " 
                    << require_mb << "MB < 0.";
        }
        memory_byte_t max_new_capacity = 
            std::max(ResourceManager::GetFreeMemoryMB(sta::DeviceManager::GetCurrentDevice(), false), 
                     0.0) * 1_MB
            + cold_model_cache->GetCachedNbytes(cold_cache_lock) 
            + total_storage_nbytes;
        
        if (max_new_capacity < new_capacity) {
          LOG(INFO) << "[MaybeAdjustTrainAndCache] require " 
                    << require_mb << "MB, but no enough memory. New capacity: " 
                    << sta::PrintByte(new_capacity) << " -> " 
                    << sta::PrintByte(max_new_capacity);
          new_capacity = std::max(
              max_new_capacity, 
              cold_model_cache->GetCacheCapacity(cold_cache_lock));
        }
        cold_model_cache->SetNewCapacity(new_capacity, cold_cache_lock);
      } else {
        // DEBUG
        auto evict_models = cold_model_cache->GetEvictModels(0, {this, nullptr}, cold_cache_lock);
        for (auto &&[name, cached_groups_id] : evict_models) {
          InferModelStore::Get()->GetModel(name)->ClearColdCache(
              cached_groups_id, rank, cold_cache_lock);
        }
        LOG(INFO) << "[MaybeAdjustTrainAndCache] require " 
                  << require_mb << "MB, but no adjust plan";
        cold_model_cache->SetNewCapacity(total_storage_nbytes, cold_cache_lock);
      }

    }
    // ensure ok
    LOG(INFO) << "[MaybeAdjustTrainAndCache] After maybe adjust, "
              << " Curr capacity: " 
              << sta::PrintByte(cold_model_cache->GetCacheCapacity(cold_cache_lock))
              << " Curr cached: " 
              << sta::PrintByte(cold_model_cache->GetCachedNbytes(cold_cache_lock))
              << " Total storage: " 
              << sta::PrintByte(total_storage_nbytes);
    CHECK_GE(cold_model_cache->GetCacheCapacity(cold_cache_lock), total_storage_nbytes);
    
    if (cold_model_cache->TakeSpace(total_storage_nbytes)) {
      LOG(INFO) << "[MaybeAdjustTrainAndCache] Secound take space success, new capacity " 
                << sta::PrintByte(cold_model_cache->GetCacheCapacity(cold_cache_lock))
                << " new cache " << sta::PrintByte(cold_model_cache->GetCachedNbytes(cold_cache_lock));
    } else {
      memory_byte_t init_cached_nbytes = cold_model_cache->GetCachedNbytes(cold_cache_lock);
      CHECK_GE(cold_model_cache->GetCacheCapacity(cold_cache_lock), 
               cold_model_cache->GetCachedNbytes(cold_cache_lock));

      CHECK_GE(cold_model_cache->GetCachedNbytes(cold_cache_lock) + total_storage_nbytes, 
               cold_model_cache->GetCacheCapacity(cold_cache_lock));
      memory_byte_t evict_to_nbytes = cold_model_cache->GetCacheCapacity(cold_cache_lock) 
                                      - total_storage_nbytes;

      LOG(INFO) << "Current capacity: " 
                << sta::PrintByte(cold_model_cache->GetCacheCapacity(cold_cache_lock))
                << " Current cached: " 
                << sta::PrintByte(cold_model_cache->GetCachedNbytes(cold_cache_lock))
                << " Total storage: " << sta::PrintByte(total_storage_nbytes)
                << " Evict to: " << sta::PrintByte(evict_to_nbytes);

      auto evict_models = cold_model_cache->GetEvictModels(
          evict_to_nbytes, {this, nullptr}, cold_cache_lock);

      LOG(INFO) << "Current capacity: " 
                << sta::PrintByte(cold_model_cache->GetCacheCapacity(cold_cache_lock))
                << " Current cached: " 
                << sta::PrintByte(cold_model_cache->GetCachedNbytes(cold_cache_lock))
                << " Total storage: " << sta::PrintByte(total_storage_nbytes)
                << " Evict to: " << sta::PrintByte(evict_to_nbytes);
                
      size_t evict_storage = 0;
      for (auto &&[name, cached_groups_id] : evict_models) {
        evict_storage += InferModelStore::Get()->GetModel(name)->GetMemoryNbytes(0);
        InferModelStore::Get()->GetModel(name)->ClearColdCache(
            cached_groups_id, rank, cold_cache_lock);
      }

      size_t release_nbytes = init_cached_nbytes 
                              - cold_model_cache->GetCachedNbytes(cold_cache_lock);
      LOG(INFO) << "[MaybeAdjustTrainAndCache] After maybe evict, "
                << " Curr capacity: " 
                << sta::PrintByte(cold_model_cache->GetCacheCapacity(cold_cache_lock))
                << " Curr cached: " 
                << sta::PrintByte(cold_model_cache->GetCachedNbytes(cold_cache_lock))
                << " Total storage: " << sta::PrintByte(total_storage_nbytes)
                << " Evict: " 
                << sta::PrintByte(evict_storage) << " | " 
                << sta::PrintByte(release_nbytes) << ".";
      // CHECK_GE(evict_to_nbytes, cold_model_cache->GetCachedNbytes(cold_cache_lock) + total_storage_nbytes);
      cold_model_cache->SetNewCapacity(
          cold_model_cache->GetCacheCapacity(cold_cache_lock) - total_storage_nbytes, 
          cold_cache_lock);
    }
  } /* end if: need adjust train / infer cache */
  cold_model_cache->UnblockProfilter();
 
#if 1
  // directly release memory changing lock, 
  // infer may alloc memory but without this lock
  return false;
#else
  // delay release the lock, will be released after infer finish alloc
  return true;
#endif
}

bool Model::SetupMemory(size_t rank, 
                        std::unique_lock<std::mutex> &cold_cache_lock, 
                        std::unique_lock<std::mutex> &model_lock) {
  CHECK(status_[rank] == Status::kWithoutMemory);

  ColdModelCache::Get(device_.device_id)->PopCacheItem(
      name_, rank, true, cold_cache_lock);
  
  if (Config::IsColocateMode() && Config::ondemand_adjust) {
    PROFILE_START(InferAdjustAlloc);
    if (!Config::colocate_config.skip_malloc) {
      MaybeAdjustTrainAndCache(rank, cold_cache_lock, model_lock);
    }
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
  auto warm_model_cache = WarmModelCache::Get(device_.device_id);
  auto cold_model_cache = ColdModelCache::Get(device_.device_id);
  num_worker_.fetch_add(1, std::memory_order_relaxed);

  if (barrier != nullptr) {
    int err = pthread_barrier_wait(barrier);
    CHECK(err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD) << "err: " << err << ".";
  } 

  while (!InferModelStore::Initialized()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  
  {
    auto reserved_lock = warm_model_cache->ReserveCache(name_, rank);
    auto cold_cache_lock = cold_model_cache->Lock();
    CHECK(status_[rank] == Status::kWithoutMemory);
    executors_[rank]->Init(true);
    ChangeStatus(rank, Status::kReady);
  }
  DLOG(INFO) << "[Model Inference] " << name_ << " (rank " << rank << ") Init Success";

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

    // avoid interference
    if (Config::dynamic_sm_partition) {
      WaitEstimateTPC();
    }

    // let cache serve models in a fifo manner
    auto reserve_cache_begin = Profiler::Now();
    auto reserved_lock = warm_model_cache->OrderedReserveCache(name_, rank, jobs);
    auto reserve_cache_ms = Profiler::MilliFrom(reserve_cache_begin);

    // [switch mode] before infering, first claim infering execution
    InferModelStore::InferingInc(tvm_graph_.get(), executors_[rank].get());

    // lock to avoid be interrupted by memory reclaim
    auto cold_cache_lock = cold_model_cache->Lock();
    DLOG(INFO) << "[Model Inference] " << name_ 
               << " (rank " << rank << ") Acquire cold cache lock success.";
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

    // claim gpu sm
    int recorded_required_tpc_num = required_num_tpc_;
    if (Config::dynamic_sm_partition && recorded_required_tpc_num > 0) {
      SMPartitioner::Get(device_.device_id)
          ->AddInferRequiredTpcNum(recorded_required_tpc_num);
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

    if (Config::dynamic_sm_partition && recorded_required_tpc_num > 0) {
      SMPartitioner::Get(device_.device_id)
          ->DecInferRequiredTpcNum(recorded_required_tpc_num);
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

    ss << " total_infer_ms=" << Profiler::Milli(infer_begin, infer_end);
    if (WarmModelCache::Enable()) { ss << " | reserve_cache_ms=" << reserve_cache_ms; }
    LOG_IF(INFO, Config::log_infer_time) << ss.str();

    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferRealBatchSize, jobs.size());
    for (auto& job : jobs) {
      job->RecordFinished();
      job->RecordProfile();
      auto data = job->GetInferData();
      data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    }
    InferModelStore::InferingDec(tvm_graph_.get(), executors_[rank].get());
    Profiler::InferRespInc(jobs.size());

    // estimate tpc after first infer
    // to avoid interference w/ training, do this during infer warmup
    if (Config::dynamic_sm_partition 
        && Config::estimate_infer_model_tpc 
        && required_num_tpc_ == -1) {
      EstimateTPC(rank, *graph_executor);
    }
  }

  std::stringstream exit_log_ss;
  exit_log_ss << "[Model Inference] model " << tvm_graph_->GetModelRank() 
              << " worker " << rank << " exit";

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

  CHECK_EQ(batch_shape.size(), input_info_[input_id].first.size()) 
      << "input shape dimension mismatch";
  CHECK_LE(batch_shape[0], input_info_[input_id].first[0]) 
      << "out of model batch size";

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
        input_host_buf, const_cast<DLTensor*>(input_dev), 
        graph_executor.GetExecStream());
    ::tvm::runtime::DeviceAPI::Get(device_)->StreamSync(
        device_, graph_executor.GetExecStream());
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
        output_dev, const_cast<DLTensor*>(output_host_buf),
        graph_executor.GetExecStream());
    ::tvm::runtime::DeviceAPI::Get(device_)->StreamSync(
        device_, graph_executor.GetExecStream());
  } else {
    LOG(FATAL) << "unsupport device type " << device_.device_type;
  }

  CHECK_LE(jobs.size(), static_cast<size_t>(output_host_buf->shape[0])) 
      << "out of model batch size";
  std::vector<int64_t> shape{output_host_buf->shape, 
                             output_host_buf->shape + output_host_buf->ndim};
  shape[0] = 1;

  size_t offset = 0;
  size_t output_nbytes = 
      ::tvm::runtime::GetDataSize(*output_host_buf) / output_host_buf->shape[0];
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

void Model::EstimateTPC(uint32_t rank, tvm::Executor &graph_executor) {
  CHECK(Config::dynamic_sm_partition);

  if (required_num_tpc_ != -1) {
    return; // skip already estimated
  }

  std::unique_lock lock{Model::estimate_tpc_mut_};

  static std::unordered_map<std::string, int> tpc_map;
  std::string model_name_without_dup_id = GetModelNameWithoutDuplicatedId(name_);
  if (tpc_map.count(model_name_without_dup_id) > 0) {
    required_num_tpc_ = tpc_map[model_name_without_dup_id];
    return;
  }

  Model::estimating_tpc_ = true;
  // LOG(INFO) << name_ << " begin estimate tpc " << Model::estimating_tpc_;
  while (InferModelStore::GetNumInferingModel() > 1) {
    // LOG(INFO) << "wait other model finish infering";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  CHECK(status_[rank] == Status::kReady);
  
  auto get_exec_ms = [&]() {
    int repeat = 100;
    auto t0 = Profiler::Now(); 
    for (int i = 0; i < repeat; i++)
      graph_executor.Run(); 
    return Profiler::MilliFrom(t0) / repeat;
  };

  double std_exec_ms = get_exec_ms();
  double target_exec_ms = std_exec_ms * Config::infer_exec_time_estimate_scale;

  // auto num_tot_sm = GetGPUNumSM(0);
  // auto num_tot_tpc = num_tot_sm >> 1;
  auto num_tot_tpc = SMPartitioner::Get(device_.device_id)->GetGPUNumTpc();
  int left = 0, right = num_tot_tpc + 1;
  while (left + 1 < right) {
    int mid = left + (right - left) / 2;
    uint64_t mask_64 = -1;
    mask_64 = mask_64 << mid;
    SMPartitioner::Get(device_.device_id)
        ->SetStreamTpcMask(static_cast<cudaStream_t>(graph_executor.GetExecStream()), mask_64);
    DLOG(INFO) << SMPartitioner::Get(device_.device_id)
                      ->CheckStreamSM(static_cast<cudaStream_t>(graph_executor.GetExecStream()));
    double exec_ms = get_exec_ms();
    if (exec_ms >= target_exec_ms) {
      left = mid;
    } else {
      right = mid;
    }
    DLOG(INFO) << "[EstimateTPC] " <<  model_name_without_dup_id 
              << " num_sm " << (mid << 1) << " exec_ms " << exec_ms << " / " << target_exec_ms;
  }
  required_num_tpc_ = left;
  LOG(INFO) << "[Model TPC Estimate] " << name_ << " required num_tpc " << required_num_tpc_;
  tpc_map[model_name_without_dup_id] = required_num_tpc_;
  
  Model::estimating_tpc_ = false;
  // LOG(INFO) << name_ << " notify estimate tpc " << Model::estimating_tpc_;
  Model::estimate_tpc_cv_.notify_all();
}

void Model::WaitEstimateTPC() {
  if (Model::estimating_tpc_) {
    // LOG(INFO) << name_ << " wait estimate tpc 1 " << Model::estimating_tpc_;
    std::unique_lock<std::mutex> lock{Model::estimate_tpc_mut_};
    // LOG(INFO) << name_ << " wait estimate tpc 2 " << Model::estimating_tpc_;
    Model::estimate_tpc_cv_.wait(lock, [&]() { return !Model::estimating_tpc_; });
    // LOG(INFO) << name_ << " wait estimate tpc done";
  }
}

}