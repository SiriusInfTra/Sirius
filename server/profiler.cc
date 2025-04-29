#include <server/logging_as_glog.h>
#include <server/model_store/infer_model_store.h>
#include <server/model_store/model_cache.h>
#include <server/train_launcher.h>
#include <server/profiler.h>
#include <server/config.h>
#include <server/control/controller.h>

#include <common/cuda_allocator.h>
#include <common/util.h>
#include <common/device_manager.h>
#include <common/sm_partition.h>
#include <common/inf_tra_comm/communicator.h>

#include <boost/format.hpp>
#include <boost/range/irange.hpp>
#include <numeric>
#include <regex>
#include <limits>


namespace colserve {
namespace {
std::string GetMemMbStr(memory_byte_t bytes) {
  return (boost::format("%.1f Mb") % (1.0 * bytes / 1024 / 1024)).str();
}
std::string GetMemMbStr(memory_mb_t mb) { 
  return (boost::format("%.1f Mb") % mb).str();
}

void FmtTable(std::ostream &os, const std::vector<std::vector<std::string>> &table) {
  std::vector<size_t> col_widths(table[0].size(), 0);
  for (auto &row : table) {
    for (size_t i = 0; i < row.size(); i++) {
      col_widths[i] = std::max(col_widths[i], row[i].size());
    }
  }
  for (auto &row : table) {
    for (size_t i = 0; i < row.size(); i++) {
      os << std::setw(col_widths[i]) << row[i] << ", ";
    }
    os << std::endl;
  }
}

template<typename T>
double mean(const std::vector<T> &vec) {
  if (vec.empty()) return std::numeric_limits<double>::quiet_NaN();
  return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

int GetDcgmFieldValues(dcgm_field_entity_group_t entity_grp_id,
                       dcgm_field_eid_t entity_id,
                       dcgmFieldValue_v1 *values,
                       int num_values, void* user_data) {
  auto& entry_stat = *static_cast<Profiler::dcgmEntityStat*>(user_data);

  CHECK(entity_grp_id == DCGM_FE_GPU)
      << "entity_grp_id " << entity_grp_id << " entity_id " << entity_id;

  int device_id = sta::DeviceManager::GetVisibleGpuId(entity_id);
  for (int i = 0; i < num_values; i++) {
    dcgm_field_meta_p field = DcgmFieldGetById(values[i].fieldId);
    if (field == nullptr) {
      LOG(FATAL) << "unknown field " << values[i].fieldId;
    }
    // LOG(INFO) << "field " << field->fieldId << " " << field->fieldType;
    switch (field->fieldType) {
      case DCGM_FT_DOUBLE:
        entry_stat[device_id][field->fieldId] = values->value.dbl;
        break;
      case DCGM_FT_INT64:
        entry_stat[device_id][field->fieldId] = values->value.i64;
        break;
      default:
        LOG(FATAL) << "unsupport DCGM field type";
    }
  }
  return 0;
}
}

#define LOG_ITEM(enum, item) case enum::item: os << #item; return os;

std::ostream& operator<<(std::ostream &os, Profiler::EventItem item) {
  switch (item)
  {
    LOG_ITEM(Profiler::EventItem, TrainAdjustStart)
    LOG_ITEM(Profiler::EventItem, TrainAdjustEnd)
    LOG_ITEM(Profiler::EventItem, InferAllocStorageStart)
    LOG_ITEM(Profiler::EventItem, InferAllocStorageEnd)
    LOG_ITEM(Profiler::EventItem, InferAdjustAllocStart)
    LOG_ITEM(Profiler::EventItem, InferAdjustAllocEnd)
    LOG_ITEM(Profiler::EventItem, InferLoadParamStart)
    LOG_ITEM(Profiler::EventItem, InferLoadParamEnd)
    LOG_ITEM(Profiler::EventItem, AddInfer)
    LOG_ITEM(Profiler::EventItem, InferExit)
  default:
    return os;
  }
}

std::ostream& operator<<(std::ostream &os, Profiler::PerfItem item) {
  switch (item) {
    LOG_ITEM(Profiler::PerfItem, InferJobQueue)
    LOG_ITEM(Profiler::PerfItem, InferJobProcess)

    LOG_ITEM(Profiler::PerfItem, InferSetInput)
    LOG_ITEM(Profiler::PerfItem, InferExec)
    LOG_ITEM(Profiler::PerfItem, InferGetOutput)

    LOG_ITEM(Profiler::PerfItem, TrainAdjust)
    LOG_ITEM(Profiler::PerfItem, TrainFirstAdjust)
    LOG_ITEM(Profiler::PerfItem, InferAllocStorage)
    LOG_ITEM(Profiler::PerfItem, InferAdjustAlloc)
    LOG_ITEM(Profiler::PerfItem, InferLoadParam)
    LOG_ITEM(Profiler::PerfItem, InferPipelineExec)

    LOG_ITEM(Profiler::PerfItem, InferNumModelOnSwitch)

    LOG_ITEM(Profiler::PerfItem, InferRealBatchSize)
    LOG_ITEM(Profiler::PerfItem, InferModelLoad)
    LOG_ITEM(Profiler::PerfItem, InferModelColdCacheHit)

    LOG_ITEM(Profiler::PerfItem, LLMNumPromptTokens)
    LOG_ITEM(Profiler::PerfItem, LLMNumGenTokens)
    LOG_ITEM(Profiler::PerfItem, LLMBackendQueue)
    LOG_ITEM(Profiler::PerfItem, LLMPrefill)
    LOG_ITEM(Profiler::PerfItem, LLMTimeBetweenTokens)

    LOG_ITEM(Profiler::PerfItem, LLMReclaimKVCacheBlkGrp)
    LOG_ITEM(Profiler::PerfItem, LLMAllocKVCacheBlkGrp)

    LOG_ITEM(Profiler::PerfItem, LLMKVCacheMap)
    LOG_ITEM(Profiler::PerfItem, LLMKVCacheUnMap)
  default:
    return os;
  }
}

std::unique_ptr<Profiler> Profiler::profiler_;

std::pair<size_t, size_t> Profiler::GetGPUMemInfo() {
  size_t free, total;
  COL_CUDA_CALL(cudaMemGetInfo(&free, &total));
  return {free, total};
}

size_t Profiler::GetLastInferMem(int device_id) {
  CHECK(profiler_ != nullptr);
  return profiler_->last_infer_mems_[device_id];
}

size_t Profiler::GetLastTrainMem(int device_id) {
  CHECK(profiler_ != nullptr);
  return profiler_->last_train_mems_[device_id];
}

void Profiler::Init(const std::string &profile_log_path) {
  CHECK(profiler_ == nullptr);
  profiler_ = std::make_unique<Profiler>(profile_log_path);
}

void Profiler::Start() {
  std::unique_lock infer_info_lock{profiler_->infer_info_mut_};
  // std::unique_lock event_info_lock{profiler_->event_info_mut_};
  profiler_->infer_info_.clear();
  // profiler_->event_info_.clear();
  for (int i = 0; i < sta::DeviceManager::GetNumVisibleGpu(); i++) {
    profiler_->resource_infos_[i].clear();
  }
  profiler_->infering_memory_nbytes_.clear();
  profiler_->start_profile_ = true;
}

void Profiler::Shutdown() {
  profiler_->thread_->join();
  if (Config::profile_gpu_smact) {
    COL_DCGM_CALL(dcgmShutdown());
  }
  profiler_->WriteLog();
}

Profiler* Profiler::Get() {
  CHECK(profiler_ != nullptr);
  return profiler_.get();
}

void Profiler::InferReqInc(uint64_t x) {
  CHECK(profiler_ != nullptr);
  std::unique_lock lock{profiler_->infer_stat_mut_};
  profiler_->infer_stat_.num_requests += x;
  ctrl::Controller::Get()->SetInferStatus(ctrl::InferStatus::kRunning);
}

void Profiler::InferRespInc(uint64_t x) {
  CHECK(profiler_ != nullptr);
  std::unique_lock lock{profiler_->infer_stat_mut_};
  profiler_->infer_stat_.num_response += x;
  if (profiler_->infer_stat_.num_response == profiler_->infer_stat_.num_requests) {
    ctrl::Controller::Get()->SetInferStatus(ctrl::InferStatus::kIdle);
  }
  // LOG(INFO) << "infer resp " << profiler_->infer_stat_.num_response
  //           << " req " << profiler_->infer_stat_.num_requests;
}

Profiler::Profiler(const std::string &profile_log_path)
    : profile_log_path_(profile_log_path), start_profile_(false) {
  if (Config::profile_gpu_smact) {
    COL_DCGM_CALL(dcgmInit());
    char ip_addr[] = "127.0.0.1";
    COL_DCGM_CALL(dcgmConnect(ip_addr, &dcgm_handle_));
  }
  stp_ = GetTimeStamp();

  uint32_t dev_cnt;
  COL_NVML_CALL(nvmlDeviceGetCount_v2(&dev_cnt));
  CHECK_GT(dev_cnt, 0);

  std::array<nvmlDevice_t, MAX_DEVICE_NUM> devices{0};
  CHECK_GE(sta::DeviceManager::GetNumVisibleGpu(), 1);
  for (int i = 0; i < sta::DeviceManager::GetNumVisibleGpu(); i++) {
    auto gpu_uuid = sta::DeviceManager::GetGpuSystemUuid(i);
    COL_NVML_CALL(nvmlDeviceGetHandleByUUID(gpu_uuid.c_str(), &devices[i]));
  }

  nvmlDevice_t device;
  auto gpu_uuid = sta::DeviceManager::GetGpuSystemUuid(0);
  COL_NVML_CALL(nvmlDeviceGetHandleByUUID(gpu_uuid.c_str(), &device));

  // CHECK MPS
  if (colserve::Config::check_mps && !colserve::Config::no_infer) {
    uint32_t info_cnt = 0;
#if !defined(USE_NVML_V3) || USE_NVML_V3 != 0
    auto nvml_err = nvmlDeviceGetMPSComputeRunningProcesses_v3(
        devices[0], &info_cnt, NULL);
    if (nvml_err == NVML_SUCCESS && info_cnt == 0) {
      LOG(FATAL) << "MPS is not enabled, please start MPS server by nvidia-cuda-mps-control";
    }
#endif
  }

#if defined(USE_NVML_V3) && USE_NVML_V3 == 0
  LOG(FATAL) << "USE_NVML_V3 is set to 0, profiler will not record memory info, "
             << "mps server will not be checked";
#endif

  thread_ = std::make_unique<std::thread>(&Profiler::ProfileThread, this, devices);
}

Profiler::~Profiler() {
}

void Profiler::RecordEvent(EventItem item) {
  std::unique_lock lock{event_info_mut_};
  event_info_.push_back({Profiler::GetTimeStamp(), item});
}

void Profiler::RecordEvent(EventItem item, Profiler::time_point_t tp) {
  std::unique_lock lock{event_info_mut_};
  event_info_.push_back({Profiler::GetTimeStamp(), item});
}

void Profiler::RecordPerf(PerfItem item, double value) {
  auto key = static_cast<int>(item);
  std::unique_lock lock{perf_info_mut_};
  perf_info_[key].push_back({Profiler::GetTimeStamp(), value});
}

void Profiler::RecordPerf(PerfItem item, 
                          Profiler::time_point_t start, 
                          Profiler::time_point_t end) {
  auto value = std::chrono::duration<double, std::milli>(end - start).count();
  RecordPerf(item, value);
}

void Profiler::SetWorkloadStartTimeStamp(long ts, double delay_before_profile) {
  workload_start_time_stamp_ = ts;
  delay_before_profile_ = delay_before_profile;
  LOG(INFO) << "workload start at " << ts 
            << ", profile will start after " << delay_before_profile
            << " sec from this time point";
}

void Profiler::SetWorkloadEndTimeStamp(long ts) {
  workload_end_time_stamp_ = ts; 
  LOG(INFO) << "workload end at " << ts;
}

void Profiler::ProfileThread(std::array<nvmlDevice_t, MAX_DEVICE_NUM> devices) {
  while (!this->start_profile_) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  LOG(INFO) << "start profiler thread";
  uint32_t pid = getpid();

  // setup DCMG for monitor computing resource
  if (Config::profile_gpu_smact || Config::profile_gpu_util) {
    char dcgm_group_name[128]; // = "dcgm_colserve";
    snprintf(dcgm_group_name, sizeof(dcgm_group_name), "dcgm_colserve_%d", pid);
    COL_DCGM_CALL(dcgmGroupCreate(this->dcgm_handle_, DCGM_GROUP_EMPTY, 
                              dcgm_group_name, &this->dcgm_gpu_grp_));
    for (int i = 0; i < sta::DeviceManager::GetNumVisibleGpu(); i++) {
      COL_DCGM_CALL(dcgmGroupAddEntity(this->dcgm_handle_, 
                                       this->dcgm_gpu_grp_, 
                                       DCGM_FE_GPU, 
                                       sta::DeviceManager::GetGpuSystemId(i)));
    }
    std::vector<uint16_t> field_ids;
    if (Config::profile_gpu_util) {
      field_ids.push_back(DCGM_FI_DEV_GPU_UTIL);
    }
    if (Config::profile_gpu_smact) {
      field_ids.push_back(DCGM_FI_PROF_SM_ACTIVE);
    }
    COL_DCGM_CALL(dcgmFieldGroupCreate(this->dcgm_handle_, field_ids.size(), 
                                       field_ids.data(), dcgm_group_name, 
                                       &this->dcgm_field_grp_));
    auto micro_delay = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::milliseconds(this->monitor_interval_ms_)).count();
    COL_DCGM_CALL(dcgmWatchFields(this->dcgm_handle_, this->dcgm_gpu_grp_,
                                  this->dcgm_field_grp_, micro_delay,
                                  0, 2));
    COL_DCGM_CALL(dcgmUpdateAllFields(this->dcgm_handle_, 1));
  }

  // main profile loop
  while (Config::running) {
    std::array<ResourceInfo, MAX_DEVICE_NUM> cur_resource_infos{0};

    CollectMemoryResourceInfo(devices, cur_resource_infos);
    CollectComputingResourcesInfo(devices, cur_resource_infos);
    
    int num_device = sta::DeviceManager::GetNumVisibleGpu();
    for (int device_id = 0; device_id < num_device; device_id++) {
      resource_infos_[device_id].emplace_back(Profiler::GetTimeStamp(), 
                                              cur_resource_infos[device_id]);
    }

    if (!Config::serving_llm) {
      infering_memory_nbytes_.emplace_back(
          Profiler::GetTimeStamp(), 
          InferModelStore::GetInferingModelNbytes());
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(monitor_interval_ms_));
  }
}

void Profiler::CollectMemoryResourceInfo(
      const std::array<nvmlDevice_t, MAX_DEVICE_NUM> &devices,
      std::array<ResourceInfo, MAX_DEVICE_NUM> &res_infos) {
  if (Config::no_infer) {
    return;
  }
  constexpr uint32_t max_proc_info_cnt = 32;
  nvmlProcessInfo_t proc_infos[32];

  for (auto device_id : boost::irange(sta::DeviceManager::GetNumVisibleGpu())) {
    auto &res_info = res_infos[device_id];
    if (!Config::use_shared_tensor || !Config::use_shared_tensor_train) {
      uint32_t proc_info_cnt = max_proc_info_cnt;
      COL_NVML_CALL(nvmlDeviceGetComputeRunningProcesses_v3(
          devices[device_id], &proc_info_cnt, proc_infos));
      for (uint32_t i = 0; i < proc_info_cnt; i++) {
        if (!Config::use_shared_tensor_train && proc_infos[i].pid == getpid()) {
          res_info.infer_mem = proc_infos[i].usedGpuMemory;
        // } else if (proc_infos[i].pid == TrainLauncher::Get()->GetTrainPid()) {
        } else if (proc_infos[i].pid == ctrl::InfTraCommunicator::GetTrainPIDNotCheckValid(device_id)) {
          res_info.train_mem = proc_infos[i].usedGpuMemory;
          res_info.train_all_mem = res_info.train_mem;
        }
      }
      
      size_t free, total;
      // COL_CUDA_CALL(cudaSetDevice(device_id));
      COL_CUDA_CALL(cudaMemGetInfo(&free, &total));
      res_info.gpu_used_mem = total - free;
      if (Config::use_shared_tensor) {
        res_info.infer_mem = sta::CUDAMemPool::Get(device_id)->InferMemUsage();
      }
    } else {
      auto read_cold_cache_info = [&]() {
        auto [cap, cache] = ColdModelCache::Get(device_id)->GetColdCacheCapacityAndCache();
        if (Config::cold_cache_max_capability_nbytes != 0) {
          res_info.cold_cache_nbytes = cache;
          
          res_info.cold_cache_buffer_mb = 
              ColdModelCache::Get(device_id)->GetBufferMBUnsafe();
         
          res_info.infer_mem_in_cold_cache_buffer_mb = 
              ColdModelCache::Get(device_id)->GetColdCacheReleasableMemoryMBUnsafe();
          res_info.cold_cache_size_mb = sta::ByteToMB(cap);
        }
      };

      auto read_resource_info = [&]() {
        if (!Config::serving_llm) {
          res_info.infer_mem = sta::CUDAMemPool::Get(device_id)->InferMemUsage();
        } else {
          res_info.infer_mem = sta::CUDAMemPool::Get(device_id)->InferMemUsageByPage();
        }
        res_info.train_mem = sta::CUDAMemPool::Get(device_id)->TrainMemUsage();
        res_info.train_all_mem = 
            sta::CUDAMemPool::Get(device_id)->TrainAllMemUsage();
        res_info.gpu_used_mem = 
            static_cast<size_t>(Config::cuda_memory_pool_gb * 1_GB);
        
        if (!Config::serving_llm) {
          read_cold_cache_info();
        }
      };

      if (Config::profiler_acquire_resource_lock) {
        // as all model require/release memory will get the cold cache
        // lock before do the operation, so we only need to get the lock
        // to ensure atomic read of the resource info
        auto cold_cache_lock = ColdModelCache::Get(device_id)->Lock();
        read_resource_info();
      } else {
        read_resource_info();
      }
    }

    this->last_infer_mems_[device_id] = res_info.infer_mem;
    this->last_train_mems_[device_id] = res_info.train_mem;
  }
}

void Profiler::CollectComputingResourcesInfo(
      const std::array<nvmlDevice_t, MAX_DEVICE_NUM> &devices,
      std::array<ResourceInfo, MAX_DEVICE_NUM> &res_infos) {
  if (Config::serving_llm) {
    return;
  }

  int num_device = sta::DeviceManager::GetNumVisibleGpu();
  for (int device_id = 0; device_id < num_device; device_id++) {
    auto &res_info = res_infos[device_id];
    if (Config::dynamic_sm_partition && Config::profile_sm_partition) {
      res_info.infer_required_tpc_num = 
          SMPartitioner::Get(device_id)->GetInferRequiredTpcNum();
      res_info.train_avail_tpc_num = 
          SMPartitioner::Get(device_id)->GetTrainAvailTpcNum();
    }
  }

  if (Config::profile_gpu_util || Config::profile_gpu_smact) {
    Profiler::dcgmEntityStat dcgm_stat;
    COL_DCGM_CALL(dcgmGetLatestValues_v2(this->dcgm_handle_, 
                                         this->dcgm_gpu_grp_, 
                                         this->dcgm_field_grp_, 
                                         &GetDcgmFieldValues, 
                                         &dcgm_stat));
    for (int device_id = 0; device_id < num_device; device_id++) {
      auto & device_dcgm_stat = dcgm_stat[device_id];
      if (Config::profile_gpu_util) {
        if (auto iter = device_dcgm_stat.find(DCGM_FI_DEV_GPU_UTIL); 
            iter != device_dcgm_stat.end()) {
          res_infos[device_id].gpu_util = iter->second;
        } else {
          res_infos[device_id].gpu_util = std::numeric_limits<double>::quiet_NaN();
        }
      }
      if (Config::profile_gpu_smact) {
        if (auto iter = device_dcgm_stat.find(DCGM_FI_PROF_SM_ACTIVE); 
            iter != device_dcgm_stat.end()) {
          res_infos[device_id].sm_activity = iter->second;
        } else {
          res_infos[device_id].sm_activity = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }
  }
}


void Profiler::WriteLog() {
  std::ofstream ofs{profile_log_path_};

  double infer_avg_exec = -1;
  
  ofs << std::fixed << std::setprecision(1)
      << "[Perf Info] workload start time stamp " << workload_start_time_stamp_ 
      << " delay before profile " << delay_before_profile_ << " sec, " 
      << " workload end time stamp " << workload_end_time_stamp_ << std::endl;

  long workload_profile_start_ts = workload_start_time_stamp_ 
                                   + static_cast<long>(delay_before_profile_ * 1000);
  long workload_profile_end_ts = workload_end_time_stamp_;


  std::vector<int> sorted_perf_info_keys;
  for (auto &it : perf_info_) {
    sorted_perf_info_keys.push_back(it.first);
  }
  std::sort(sorted_perf_info_keys.begin(), sorted_perf_info_keys.end());
  for (auto key : sorted_perf_info_keys) {
    auto item = static_cast<Profiler::PerfItem>(key);
    std::vector<double> item_perf_info;
    for (auto &p : perf_info_[key]) {
      auto time_stamp = std::get<0>(p);
      auto value = std::get<1>(p);
      if (time_stamp > workload_start_time_stamp_ + static_cast<long>(delay_before_profile_ * 1000)) {
        item_perf_info.push_back(value);
      }
    }
    if (item_perf_info.empty()) {
      ofs << item << ": no record after workload start profile time stamp" << std::endl;
      continue;
    }
    auto max = *std::max_element(item_perf_info.begin(), item_perf_info.end());
    auto min = *std::min_element(item_perf_info.begin(), item_perf_info.end());
    auto sum = std::accumulate(item_perf_info.begin(), item_perf_info.end(), 0.0);
    double avg = -1;
    if (item_perf_info.size() > 0) {
      avg = 1.0 * sum / item_perf_info.size();
    }
    auto sorted = item_perf_info;
    sort(sorted.begin(), sorted.end());
    ofs << static_cast<PerfItem>(key) << std::fixed << std::setprecision(1) << ":"
        << " avg " << avg << " max " << max << " min " << min << " cnt " << item_perf_info.size() << " |"
        << " p99 " << sorted[int(0.99 * sorted.size())]
        << " p95 " << sorted[int(0.95 * sorted.size())]
        << " p90 " << sorted[int(0.90 * sorted.size())]
        << " p80 " << sorted[int(0.80 * sorted.size())]
        << " p70 " << sorted[int(0.70 * sorted.size())]
        << " p60 " << sorted[int(0.60 * sorted.size())]
        << " p50 " << sorted[int(0.50 * sorted.size())]
        << std::endl;
    if (item == Profiler::PerfItem::InferExec) {
      infer_avg_exec = avg;
    }
  }
  for (int i = 0; i < static_cast<int>(Profiler::PerfItem::NumPerfItem); i++) {
    auto item = static_cast<Profiler::PerfItem>(i);
    if (perf_info_.find(i) == perf_info_.end()) {
      ofs << item << ": no record" << std::endl;
    }
  }
  ofs << std::endl;

  ofs << "[Event Info]" << std::endl;
  for (auto &e : event_info_) {
    ofs << std::get<0>(e) << ": "
        << std::get<1>(e) << ": "
        << std::endl;
  }
  ofs << std::endl;

  if (Config::profile_gpu_util || Config::profile_gpu_smact) {
    for (int device_id = 0; device_id < sta::DeviceManager::GetNumVisibleGpu(); device_id++) {
      ofs << str(boost::format("[GPU Util Info | Device %d]") % device_id) << std::endl;
      auto gpu_utils = SelectResourceInfo<decltype(((ResourceInfo*)nullptr)->gpu_util)>(
          device_id, offsetof(ResourceInfo, gpu_util), 
          workload_profile_start_ts, workload_profile_end_ts, 
          [](double v) { return !std::isnan(v); });
      auto gpu_smacts = SelectResourceInfo<decltype(((ResourceInfo*)nullptr)->sm_activity)>(
          device_id, offsetof(ResourceInfo, sm_activity), 
          workload_profile_start_ts, workload_profile_end_ts,
          [](double v) { return !std::isnan(v); });

      if (Config::profile_gpu_util) {
        ofs << "GPU Util: avg " << mean(gpu_utils) << " %"
            << std::endl;
      }
      if (Config::profile_gpu_smact) {
        ofs << "SM Activity: avg "  << mean(gpu_smacts) * 100 << " %"
            << std::endl;
      }
      ofs << std::endl;
    }
  }
  
  for (int device_id = 0; device_id < sta::DeviceManager::GetNumVisibleGpu(); device_id++) {
    ofs << str(boost::format("[Memory Info | Device %d]") % device_id) << std::endl;
    auto memory_table = FmtResourceInfos(device_id, {
      offsetof(ResourceInfo, infer_mem),
      offsetof(ResourceInfo, train_mem),
      offsetof(ResourceInfo, train_all_mem),
      offsetof(ResourceInfo, gpu_used_mem),
      offsetof(ResourceInfo, cold_cache_nbytes),
      offsetof(ResourceInfo, cold_cache_buffer_mb),
      offsetof(ResourceInfo, infer_mem_in_cold_cache_buffer_mb),
      offsetof(ResourceInfo, cold_cache_size_mb),
    }, std::nullopt, std::nullopt);
    FmtTable(ofs, memory_table);
    ofs << std::endl;
  }

  if (Config::dump_adjust_info) {
    std::vector<Profiler::PerfItem> adjust_item = {
      Profiler::PerfItem::TrainAdjust,
      Profiler::PerfItem::InferAllocStorage,
      Profiler::PerfItem::InferLoadParam,
      Profiler::PerfItem::InferPipelineExec
    };

    ofs << "[Adjust Info]" << std::endl;
    for (auto item : adjust_item) {
      // auto item = static_cast<int>(key);
      std::vector<double> item_perf_info;
      for (auto &p : perf_info_[static_cast<int>(item)]) {
        auto time_stamp = std::get<0>(p);
        auto value = std::get<1>(p);
        if (time_stamp > workload_start_time_stamp_ + static_cast<long>(delay_before_profile_ * 1000)) {
          item_perf_info.push_back(value);
        }
      }
      if (item_perf_info.empty()) {
        ofs << item << ": no record after workload start profile time stamp" << std::endl;
        continue;
      }
      ofs << item << ":\n";
      for (size_t i = 0; i < item_perf_info.size(); i++) {
        ofs << std::setw(5) << std::fixed << std::setprecision(2) 
            << item_perf_info[i] << " ";
        if ((i + 1) % 20 == 0) ofs << std::endl;
      }
      if (item_perf_info.size() % 20 != 0) ofs << std::endl;
    }
    ofs << Profiler::PerfItem::InferExec << " avg:\n";
    ofs << std::fixed << std::setprecision(1) << infer_avg_exec << std::endl;
  }
  ofs << std::endl;

  if (Config::dynamic_sm_partition && Config::profile_sm_partition) {
    ofs << "[SM Partition Info]" << std::endl;
    auto sm_part_table = FmtResourceInfos(0, {
      offsetof(ResourceInfo, infer_required_tpc_num),
      offsetof(ResourceInfo, train_avail_tpc_num),
    }, std::nullopt, std::nullopt);
    FmtTable(ofs, sm_part_table);
    ofs << std::endl;
  }

  LOG(INFO) << "[Profiler] write profile info to " << profile_log_path_;
}

std::vector<std::vector<std::string>>
Profiler::FmtResourceInfos(
    int device_id,
    const std::vector<size_t> &field_offs,
    std::optional<time_stamp_t> start, 
    std::optional<time_stamp_t> end) {
  std::vector<std::vector<std::string>> table;

  table.push_back({});
  table.back().resize(field_offs.size() + 1);

  table[0][0] = "TimeStamp";

  for (auto &r : resource_infos_[device_id]) {
    if (start.has_value() && std::get<0>(r) < start.value()) continue;
    if (end.has_value() && std::get<0>(r) > end.value()) continue;

    std::vector<std::string> row(table[0].size(), "");
    row[0] = (boost::format("%.1f") % std::get<0>(r)).str();

    for (int fid = 1; fid <= field_offs.size(); fid++) {
      std::string header;
      std::string value;

      auto off = field_offs[fid-1];
      switch (off)
      {
      case offsetof(ResourceInfo, infer_mem):
        header = "InferMem";
        value = GetMemMbStr(std::get<1>(r).infer_mem);
        break;
      case offsetof(ResourceInfo, train_mem):
        header = "TrainMem";
        value = GetMemMbStr(std::get<1>(r).train_mem);
        break;
      case offsetof(ResourceInfo, train_all_mem):
        header = "TrainAllMem";
        value = GetMemMbStr(std::get<1>(r).train_all_mem);
        break;
      case offsetof(ResourceInfo, gpu_used_mem):
        header = "TotalMem";
        value = GetMemMbStr(std::get<1>(r).gpu_used_mem);
        break;
      case offsetof(ResourceInfo, cold_cache_nbytes):
        header = "ColdCache";
        value = GetMemMbStr(std::get<1>(r).cold_cache_nbytes);
        break;
      case offsetof(ResourceInfo, cold_cache_buffer_mb):
        header = "ColdCacheBuffer";
        value = GetMemMbStr(std::get<1>(r).cold_cache_buffer_mb);
        break;
      case offsetof(ResourceInfo, infer_mem_in_cold_cache_buffer_mb):
        header = "InferInColdCacheBuffer";
        value = GetMemMbStr(std::get<1>(r).infer_mem_in_cold_cache_buffer_mb);
        break;
      case offsetof(ResourceInfo, cold_cache_size_mb):
        header = "ColdCacheSize";
        value = GetMemMbStr(std::get<1>(r).cold_cache_size_mb);
        break;
      case offsetof(ResourceInfo, gpu_util):
        header = "GPUUtil";
        value = std::to_string(std::get<1>(r).gpu_util);
        break;
      case offsetof(ResourceInfo, gpu_mem_util):
        header = "GPUMemUtil";
        value = std::to_string(std::get<1>(r).gpu_mem_util);
        break;
      case offsetof(ResourceInfo, sm_activity):
        header = "SMActivity";
        value = std::to_string(std::get<1>(r).sm_activity);
        break;
      case offsetof(ResourceInfo, infer_required_tpc_num):
        header = "InferRequiredTpc";
        value = std::to_string(std::get<1>(r).infer_required_tpc_num);
        break;
      case offsetof(ResourceInfo, train_avail_tpc_num):
        header = "TrainAvailTpc";
        value = std::to_string(std::get<1>(r).train_avail_tpc_num);
        break;
      default:
        LOG(FATAL) << "unknown field offset " << off;
      }
      if (table[0][fid].empty()) {
        table[0][fid] = header;
      }
      row[fid] = value;
    }

    table.push_back(row);
  }
  
  return table;
}


} // namespace colserve
