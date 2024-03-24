#include "logging_as_glog.h"
#include "common/tvm_allocator.h"
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>
#include <c10/core/MemoryFormat.h>

#include <common/shape_helper.h>
#include <common/mempool.h>
#include <common/util.h>
#include <server/train_launcher.h>
#include <server/infer_model_store.h>
#include <server/profiler.h>
#include <server/controller.h>
#include <server/resource_manager.h>
#include <server/config.h>
#include <server/tvm/executor.h>
#include <server/tvm/texture.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <regex>


namespace colserve {
namespace tvm {
namespace details {
inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < ::tvm::runtime::kAllocAlignment)
    return ::tvm::runtime::kAllocAlignment;
  return align;
}
}

namespace {
std::string GetModelNameWithoutDuplicatedId(const std::string &model_name) {
  std::regex r{"([a-zA-Z0-9_]+)(-[0-9]+)?"};
  std::smatch match;
  CHECK(std::regex_match(model_name, match, r)) << "model name " << model_name << " is not valid";
  CHECK_EQ(match.size(), 3);
  CHECK(!match[1].str().empty());
  return match[1].str();
}
}

Executor::Executor(TVMGraph &factory, size_t worker_id, const std::vector<DLDevice> &devs)
    : tvm_graph_(factory), infer_model_worker_id_(worker_id), devices_(devs), initialized_(false) {
  using namespace ::tvm::runtime;
  auto t0 = std::chrono::steady_clock::now();
  SetupStorage(false);
  SetupOpExecs();
  exec_stream_ = DeviceAPI::Get(devices_[0])
      ->CreateStream(devices_[0]);
  load_param_stream_ = DeviceAPI::Get(devices_[0])
      ->CreateStream(devices_[0]);

  param_ready_.resize(GetNumOfNodes());
  for (size_t i = 0; i < param_ready_.size(); i++) {
    param_ready_[i] = std::make_unique<std::atomic<bool>>(false);
  }
  param_ready_events_.resize(GetNumOfNodes());
  param_ready_event_ids_.resize(GetNumOfNodes(), static_cast<uint32_t>(-1));
  for (size_t i = 0; i < param_ready_events_.size(); i++) {
    CUDA_CALL(cudaEventCreate(&param_ready_events_[i]));
  }
  // pipeline_op_exec_starts_.resize(op_execs_.size());
  // pipeline_op_exec_ends_.resize(op_execs_.size());
  // for (size_t i = 0; i < op_execs_.size(); i++) {
  //   if (op_execs_[i]) {
  //     CUDA_CALL(cudaEventCreate(&pipeline_op_exec_starts_[i]));
  //     CUDA_CALL(cudaEventCreate(&pipeline_op_exec_ends_[i]));
  //   }
  // }

  if (Config::group_param_load) {
    for (auto sit = tvm_graph_.params_.begin(); sit != tvm_graph_.params_.end(); ) {
      size_t total_nbytes = 0, off = 0;
      std::vector<uint32_t> param_ids;
      auto eit = sit;
      for (; eit != tvm_graph_.params_.end() && total_nbytes < Config::group_param_load_threshold; eit++) {
        auto &p = *eit;
        auto aligned_nbytes = sta::detail::GetAlignedNbytes(GetDataSize(*p.second.operator->()));
        total_nbytes += aligned_nbytes;
        param_ids.push_back(p.first);
        param_ready_event_ids_[p.first] = param_storage_group_.size();
      }
      auto param_group = TVMArray::Empty(ShapeTuple({static_cast<int64_t>(total_nbytes)}),
         DLDataType{kDLInt, 8, 1}, DLDevice{kDLCUDAHost, 0});
      for (; sit != eit; sit++) {
        auto &p = *sit;
        std::memcpy(static_cast<char*>(param_group->data) + off, p.second->data,
          GetDataSize(*p.second.operator->()));
        auto aligned_nbytes = sta::detail::GetAlignedNbytes(GetDataSize(*p.second.operator->()));
        off += aligned_nbytes;
      }
      this->param_storage_group_.push_back(std::make_pair(param_group, param_ids));
    }
  } else {
    for (auto &p : tvm_graph_.params_) {
      param_ready_event_ids_[p.first] = p.first;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  DLOG(INFO) << "[Executor] Create " 
             << std::chrono::duration<double, std::milli>(t1-t0).count() << "ms";
}

void Executor::Init(bool load_param) {
  if (!initialized_) {   
    if (Config::IsColocateMode() && Config::ondemand_adjust) {
      PROFILE_START(InferAdjustAlloc);
      if (!Config::colocate_config.skip_malloc) AllocStorageMaybeAdjust();
      PROFILE_END(InferAdjustAlloc);
    } else {
      PROFILE_START(InferAllocStorage);
      if (!Config::colocate_config.skip_malloc) AllocStorage();
      PROFILE_END(InferAllocStorage);
    }
    
    if (!Config::colocate_config.skip_malloc)
      ReSetupDataEntry();

    if (load_param) {
      PROFILE_START(InferLoadParam);
      if (!Config::colocate_config.skip_loading) {
        LoadParams(Config::pipeline_load, Config::colocate_config.skip_malloc);
      }
      PROFILE_END(InferLoadParam);
    } else {
      // param will be loaded by pipeline
    }
    
    initialized_ = true;
  }
}

void Executor::FakeInit(bool malloc, bool load_param) {
  CHECK(!initialized_) << "FakeInit should only be called once before Init";
  if (malloc) {
    LOG(INFO) << "FakeInit malloc, skip malloc in Init";
    AllocStorage();
    ReSetupDataEntry();
  }
  if (load_param) {
    LOG(INFO) << "FakeInit load_param, skip load_param in Init";
    LoadParams(false, true);
  }
}

void Executor::DeInit() {
  CHECK(initialized_);
  if (!Config::colocate_config.skip_malloc) {
    ResetStorage();
  }
  initialized_ = false;
}

void Executor::Run() {
  using namespace ::tvm::runtime;
  CHECK(initialized_);
  DeviceAPI::Get(devices_[0])->SetStream(devices_[0], exec_stream_);
  for (size_t i = 0; i < op_execs_.size(); i++) {
    if (op_execs_[i]) op_execs_[i]();
  }
  DeviceAPI::Get(devices_[0])->StreamSync(devices_[0], exec_stream_);
}

void Executor::PipeLineLoad() {
  load_params_future_ = std::async(std::launch::async, [this]() {
    PROFILE_START(InferLoadParam);
    if (!Config::colocate_config.skip_loading) {
      LoadParams(Config::pipeline_load, Config::colocate_config.skip_malloc);
    }
    PROFILE_END(InferLoadParam);
  });
}

void Executor::PipelineRun() {
  using namespace ::tvm::runtime;
  CHECK(initialized_);
  double wait_load_ms = 0;
  double wait_ms = 0;
  double record_exec_ms = 0;

  auto begin = Profiler::Now();
  DeviceAPI::Get(devices_[0])->SetStream(devices_[0], exec_stream_);
  for (size_t i = 0; i < op_execs_.size(); i++) {
    if (op_execs_[i]) {
      auto t0 = Profiler::Now();
      // 1. wait load begin
      for (bool param_ready = false; !param_ready; ) {
        param_ready = true;
        for (auto nid : input_param_nid_[i]) {
          if (!param_ready_[nid]->load()) {
            param_ready = false;
            break;
          }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
      wait_load_ms += Profiler::MilliFrom(t0);

      auto t1 = Profiler::Now();
      // 2. wait load finish
      for (auto nid : input_param_nid_[i]) {
        auto event_id = param_ready_event_ids_[nid];
        CHECK(event_id != -1);
        CUDA_CALL(cudaStreamWaitEvent((cudaStream_t)exec_stream_, param_ready_events_[event_id]));
      }
      wait_ms += Profiler::MilliFrom(t1);

      op_execs_[i]();
      // auto t0 = std::chrono::steady_clock::now();
      // for (bool param_ready = false; !param_ready; ) {
      //   param_ready = true;
      //   for (auto nid : input_param_nid_[i]) {
      //     if (!param_ready_[nid]) {
      //       param_ready = false;
      //       break;
      //     }
      //   }
      //   std::this_thread::sleep_for(std::chrono::microseconds(10));
      // }
      // auto t1 = std::chrono::steady_clock::now();
      // wait_time += std::chrono::duration<double, std::milli>(t1-t0).count();
      // op_execs_[i]();
      // DeviceAPI::Get(tvm_graph_.devices_[0])->StreamSync(tvm_graph_.devices_[0], exec_stream_);
    }
  }
  DeviceAPI::Get(devices_[0])->StreamSync(devices_[0], exec_stream_);
  LOG(INFO) << "wait_load_ms " << wait_load_ms << " wait_ms " << wait_ms << " record_exec_ms " << record_exec_ms
            << " tot " << Profiler::MilliFrom(begin);
}

// double Executor::ComputePipelineExecTime() {
//   double ret = 0;
//   for (size_t i = 0; i < op_execs_.size(); i++) {
//     if (op_execs_[i]) {
//       float ms;
//       CUDA_CALL(cudaEventElapsedTime(&ms, pipeline_op_exec_starts_[i], pipeline_op_exec_ends_[i]));
//       ret += ms;
//     }
//   }
//   return ret;
// }

const DLTensor* Executor::GetInput(int index) const {
  CHECK(initialized_);
  CHECK_LT(static_cast<size_t>(index), tvm_graph_.input_nodes_.size());
  uint32_t eid = entry_id(tvm_graph_.input_nodes_[index], 0);
  return data_entry_[eid].operator->();
}

const DLTensor* Executor::GetInput(const std::string &index) const {
  return GetInput(GetInputIndex(index));
}

const DLTensor* Executor::GetInputHostBuf(const std::string &index) const {
  CHECK(initialized_);
  return input_cpu_pin_bufs_.at(index).operator->();
}

const DLTensor* Executor::GetOutput(int index) const {
  CHECK(initialized_);
  CHECK_LE(static_cast<size_t>(index), tvm_graph_.outputs_.size());
  uint32_t eid = entry_id(tvm_graph_.outputs_[index]);
  return data_entry_[eid].operator->();
}

const DLTensor* Executor::GetOutput(const std::string &index) const {
  return GetOutput(GetOutputIndex(index));
}

const DLTensor* Executor::GetOutputHostBuf(const std::string &index) const {
  CHECK(initialized_);
  return output_cpu_pin_bufs_.at(index).operator->();
}

uint32_t Executor::GetInputIndex(const std::string &name) const {
  auto it = tvm_graph_.input_map_.find(name);
  if (it != tvm_graph_.input_map_.end()) {
    return it->second;
  } else {
    LOG(FATAL) << "cannot find " << name << " in input";
    return -1;
  }
} 

uint32_t Executor::GetOutputIndex(const std::string &name) const {
  auto it = tvm_graph_.output_map_.find(name);
  if (it != tvm_graph_.output_map_.end()) {
    return it->second;
  } else {
    LOG(FATAL) << "cannot find " << name << " in output";
    return -1;
  }
}

// void Executor::ResetBufStorage() {
//   using namespace ::tvm::runtime;
//   for (auto &e : data_entry_) {
//     e.get_mutable()->dl_tensor.data = nullptr;
//   }
//   for (auto &s: storage_pool_) {
//     DeviceAPI::Get(s->device)->FreeDataSpace(s->device, s->data);
//     s.get_mutable()->dl_tensor.data = nullptr;
//   }
// }

// void Executor::AllocBufStorage() {
//   using namespace ::tvm::runtime;
//   for (auto &s : storage_pool_) {
//     s.get_mutable()->dl_tensor.data =
//         DeviceAPI::Get(s->device)->AllocDataSpace(
//           s->device, s->ndim, s->shape, s->dtype);
//   }
// }

void Executor::ResetStorage() {
  using namespace ::tvm::runtime;
  for (auto &p : tvm_graph_.params_) {
    param_ready_[p.first]->store(false);
  }
  for (auto & e : data_entry_) { e.DeallocToNull(); }
  for (auto & s : storage_pool_) { s.DeallocToNull(); }
  if (Config::use_shared_tensor_infer) {
    if (Config::better_alloc) { storage_group_.clear(); }
  } else {
    if (Config::infer_raw_blob_alloc) { blob_mem_.reset(); }
  }

  // for (auto &e : data_entry_) {
  //   e.get_mutable()->dl_tensor.data = nullptr;
  // }
  // for (auto &s : storage_pool_) {
  //   // DeviceAPI::Get(s->device)->FreeDataSpace(s->device, s->data);
  //   s.get_mutable()->dl_tensor.data = nullptr;
  // }
  // CHECK(cudaFree(blob_mem_) == cudaSuccess);
}

void Executor::AllocStorage() {
  using namespace ::tvm::runtime;

  auto get_storage_alloc_order = [this] () -> std::vector<uint32_t> {
    std::vector<uint32_t> storage_alloc_order;
    if (Config::group_param_load) {
      std::vector<bool> storage_record(storage_pool_.size(), false);
      for (auto & p : tvm_graph_.params_) {
        auto sid = tvm_graph_.attrs_.storage_id[p.first];
        storage_alloc_order.push_back(sid);
        storage_record[sid] = true;
      }
      for (size_t i = 0; i < storage_pool_.size(); i++) {
        if (!storage_record[i]) {
          storage_alloc_order.push_back(i);
        }
      }
    } else {
      for (size_t i = 0; i < storage_pool_.size(); i++) {
        storage_alloc_order.push_back(i);
      }
    }
    return storage_alloc_order;
  };

  if (Config::use_shared_tensor_infer) {
    if (!Config::better_alloc) {
      for (auto &s : storage_pool_) {
        s.AllocForNull(sta::MemType::kInfer);
      }
    } else {
      auto storage_alloc_order = get_storage_alloc_order();
      param_groups_nbytes_ = 0;
      if (Config::group_param_dump) {
        std::vector<size_t> model_storage_nbytes(storage_alloc_order.size());
        for (uint32_t k : storage_alloc_order) {
          auto tensor = storage_pool_[storage_alloc_order[k]];
          CHECK(tensor.IsNull());
          auto aligned_nbytes = sta::ComputeStorageNbytes(
              tensor.Shape(), tensor.Stride(), tensor->dtype, tensor.StorageOffset());
          model_storage_nbytes[k] = aligned_nbytes;
        }
        size_t model_nbytes1 = std::accumulate(model_storage_nbytes.cbegin(), model_storage_nbytes.cend(), 0U);
        std::string file_name = "nbytes_" + std::to_string(model_nbytes1) + ".txt";
        if (!std::filesystem::exists(file_name)) {
          std::ofstream handle(file_name);
          CHECK(handle.is_open());
          for (size_t nbytes : model_storage_nbytes) {
            handle << nbytes << "\n";
          }
          handle.close();
        }
      }
      size_t model_nbytes = 0, fragment_nbytes = 0;
      CHECK_GE(tvm_graph_.param_group_parti_.size(), 2);
      CHECK_EQ(tvm_graph_.param_group_parti_.front(), 0);
      CHECK_EQ(tvm_graph_.param_group_parti_.back(), storage_pool_.size());
      for (size_t k = 0; k < tvm_graph_.param_group_parti_.size() - 1; ++k) {
        size_t i = tvm_graph_.param_group_parti_[k], j = tvm_graph_.param_group_parti_[k + 1];
        size_t group_nbytes = 0;
        for (auto iter = storage_alloc_order.cbegin() + i; iter != storage_alloc_order.cbegin() + j; ++iter) {
          auto tensor = storage_pool_[*iter];
          CHECK(tensor.IsNull());
          auto aligned_nbytes = sta::ComputeStorageNbytes(
              tensor.Shape(), tensor.Stride(), tensor->dtype, tensor.StorageOffset());
          aligned_nbytes = sta::detail::GetAlignedNbytes(aligned_nbytes);
          group_nbytes += aligned_nbytes;
        }
        auto mdata_group = sta::CUDAMemPool::Get()->Alloc(
            group_nbytes, sta::MemType::kInfer, false);
        size_t off = 0;
        for (; i < j; i++) {
          auto tensor = storage_pool_[storage_alloc_order[i]];
          auto aligned_nbytes = sta::ComputeStorageNbytes(
              tensor.Shape(), tensor.Stride(), tensor->dtype, tensor.StorageOffset());
          aligned_nbytes = sta::detail::GetAlignedNbytes(aligned_nbytes);
          auto mdata = std::shared_ptr<sta::CUDAMemPool::PoolEntry>(
              new sta::CUDAMemPool::PoolEntry{static_cast<char*>(mdata_group->addr) + off, aligned_nbytes});
          tensor.SetMDataForNull(mdata);
          off += aligned_nbytes;
        }
        storage_group_.push_back(mdata_group);

        model_nbytes += group_nbytes;
        fragment_nbytes += sta::detail::AlignedNBytes<sta::TVMAllocator::ALIGN_NBYTES>(group_nbytes) - group_nbytes;
        param_groups_nbytes_ += sta::detail::AlignedNBytes<sta::TVMAllocator::ALIGN_NBYTES>(group_nbytes);
      } 
      LOG(INFO) << "[Executor] " << "internal fragment: " << sta::ByteDisplay(fragment_nbytes) << " / " << sta::ByteDisplay(model_nbytes);
    }
  } else if (Config::infer_raw_blob_alloc) {
    size_t total_nbytes = 0, off = 0;
    // constexpr size_t align = 4 * sizeof(int);
    constexpr size_t align = 1;
    static_assert(((align - 1) & align) == 0, "align must be power of 2");
    for (auto &s : storage_pool_) {
      total_nbytes += (GetDataSize(*s.operator->()) + align - 1) & (~(align - 1));
    }
    blob_mem_ = sta::CUDAMemPool::RawAlloc(total_nbytes, sta::MemType::kInfer);
    // blob_mem_ = sta::CUDAMemPool::Get()->Alloc(total_nbytes, sta::MemType::kInfer);
    for (auto &s : storage_pool_) {
      size_t nbytes = (GetDataSize(*s.operator->()) + align - 1) & (~(align - 1));
      auto mdata = std::shared_ptr<sta::CUDAMemPool::PoolEntry>(
          new sta::CUDAMemPool::PoolEntry{static_cast<char*>(blob_mem_->addr) + off, nbytes});
      s.SetMDataForNull(mdata);
      off += nbytes;
    }
  } else {
    for (auto &s: storage_pool_) {
      s.AllocForNull(sta::MemType::kInfer);
    }
  }

  // size_t total_size = 0, off = 0;
  // for (auto &s : storage_pool_) {
  //   total_size += GetDataSize(*s.operator->());
  // }
  // CHECK(cudaMalloc(&blob_mem_, total_size) == cudaSuccess);
  // for (auto &s : storage_pool_) {
  //   s.get_mutable()->dl_tensor.data = static_cast<char*>(blob_mem_) + off;
  //   off += GetDataSize(*s.operator->());
  // }

  // for (auto &s : storage_pool_) {
  //   s.get_mutable()->dl_tensor.data =
  //       DeviceAPI::Get(s->device)->AllocDataSpace(
  //         s->device, s->ndim, s->shape, s->dtype);
  // }
}

void Executor::LoadParams(bool pipeline, bool force) {
    using namespace ::tvm::runtime;
  if (force) {
    for (auto &p : tvm_graph_.params_)
      param_ready_[p.first]->store(false);
  }

  if (!Config::group_param_load) {
    for (auto &p : tvm_graph_.params_) {
      auto sid = tvm_graph_.attrs_.storage_id[p.first];
      if (!param_ready_[p.first]->load()) {
        sta::STensor storage_tensor = storage_pool_[sid];
        tvm::TVMArray::CopyFromTo(
          p.second.operator->(), storage_tensor.MutableDLTensor(), load_param_stream_);
        if (pipeline) {
          // ::tvm::runtime::DeviceAPI::Get(tvm_graph_.devices_[0])
          //     ->StreamSync(tvm_graph_.devices_[0], load_param_stream_);
          param_ready_[p.first]->store(true);
          CHECK(param_ready_event_ids_[p.first] == p.first);
          CUDA_CALL(cudaEventRecord(param_ready_events_[p.first], (cudaStream_t)load_param_stream_));
        }
      }
    }
  } else {
    // double api_1_ms = 0;
    // double api_2_ms = 0;
    // auto t_b = Profiler::Now();

    size_t sg_id = 0;
    size_t sg_off = 0;
    auto storage_group = storage_group_[sg_id++];
    for (size_t pg_id = 0; pg_id < param_storage_group_.size(); pg_id++) {
      auto & pg = param_storage_group_[pg_id];
      auto & param_group = pg.first;
      auto param_group_nbytes = static_cast<size_t>(param_group->shape[0]);
      auto & param_ids = pg.second;
      for (size_t pg_off = 0; pg_off < param_group_nbytes; ) {
        auto load_nbytes = std::min(param_group_nbytes - pg_off, storage_group->nbytes - sg_off);
        // auto t0 = Profiler::Now();
        CUDA_CALL(cudaSetDevice(devices_[0].device_id));
        CUDA_CALL(cudaMemcpyAsync(
            static_cast<char*>(storage_group->addr) + sg_off,
            static_cast<char*>(param_group->data) + pg_off,
            load_nbytes, cudaMemcpyDefault, (cudaStream_t)load_param_stream_));
        // api_1_ms += Profiler::MilliFrom(t0);
        CHECK(param_ready_event_ids_[param_ids[0]] == pg_id);
        // auto t1 = Profiler::Now();
        CUDA_CALL(cudaEventRecord(param_ready_events_[pg_id], (cudaStream_t)load_param_stream_));
        // api_2_ms += Profiler::MilliFrom(t1);
        sg_off += load_nbytes;
        pg_off += load_nbytes;
        if (sg_off == storage_group->nbytes) {
          CHECK(sg_id < storage_group_.size());
          storage_group = storage_group_[sg_id++]; sg_off = 0;
        }
      }
      if (pipeline) {
        for (auto & pid : param_ids) {
          param_ready_[pid]->store(true);
        }
      }
    }
    // auto tot_ms = Profiler::MilliFrom(t_b);
    // LOG(INFO) << "Load Params" << " api_1_ms " << api_1_ms << " api_2_ms " << api_2_ms << " tot_ms " << tot_ms;
    // LOG(INFO) << "LoadParams" << " tot_ms " << Profiler::MilliFrom(t_b);
  }

  // because we load params in async thread in pipeline mode,
  // it's ok to sync to calculate loading time
  ::tvm::runtime::DeviceAPI::Get(devices_[0])
      ->StreamSync(devices_[0], load_param_stream_);
  if (!pipeline) {
    for (auto &p : tvm_graph_.params_)
      param_ready_[p.first]->store(true);
  }
}

void Executor::ReSetupDataEntry() {
  for (size_t i = 0; i < data_entry_.size(); i++) {
    int sid = tvm_graph_.attrs_.storage_id[i];
    // TVMArray *storage;
    // if (tvm_graph_.pool_entry_[sid].params_entry) {
    //   storage = &tvm_graph_.storage_pool_[tvm_graph_.param_node_storage_id_map_[sid]];
    // } else {
    //   storage = &storage_pool_[op_node_storage_id_map_[sid]];
    // }

    // data_entry_[i].get_mutable()->dl_tensor.data =
    //     storage_pool_[sid].get_mutable()->dl_tensor.data;
    // CHECK_NE(data_entry_[i]->data, nullptr);
    auto storage_tensor = storage_pool_[sid];
    data_entry_[i].SetMDataForNull(storage_tensor.MData());
  }
}

void Executor::SetupStorage(bool alloc) {
    std::vector<DLDataType> vtype;
  for (const std::string &s_type : tvm_graph_.attrs_.dltype) {
    vtype.push_back(::tvm::runtime::String2DLDataType(s_type));
  }

  // std::vector<PoolEntry> pool_entry_;
  for (size_t i = 0; i < tvm_graph_.attrs_.shape.size(); i++) {
    int storage_id = tvm_graph_.attrs_.storage_id[i];
    std::string storage_scope = tvm_graph_.attrs_.storage_scope.empty() ? "" : tvm_graph_.attrs_.storage_scope[i];
    int device_type = static_cast<int>(devices_[0].device_type);
    if (!tvm_graph_.attrs_.device_index.empty()) {
      device_type = tvm_graph_.attrs_.device_index[i];
    }

    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entry_.size()) {
      pool_entry_.resize(sid + 1, {-1, {0}, {}});
    } else {
      CHECK_EQ(pool_entry_[sid].params_entry, false)
          << "parameter storage " << sid << " cannot be reused";
      CHECK(pool_entry_[sid].device_type == -1 || pool_entry_[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    tvm_graph_.CheckNullLinkedParam(tvm_graph_.module_, sid);
    pool_entry_[sid].param_data_entry = i;
    pool_entry_[sid].device_type = device_type;
    pool_entry_[sid].scope = storage_scope;
    if (tvm_graph_.params_.count(i)) {
      pool_entry_[sid].params_entry = true;
    }

    DLDataType t = vtype[i];
    if (!::tvm::runtime::IsTextureStorage(storage_scope)) {
      size_t size = 1;
      for (int64_t sz : tvm_graph_.attrs_.shape[i]) {
        size *= static_cast<size_t>(sz);
      }
      size_t bits = t.bits * t.lanes;
      CHECK(bits % 8U == 0U || bits == 1U || bits == 4U);
      int64_t bytes = ((bits + 7U) / 8U) * size;
      pool_entry_[sid].shape[0] = std::max(pool_entry_[sid].shape[0], bytes);
      pool_entry_[sid].dtype = DLDataType{kDLFloat, 32, 1};
    } else {
      CHECK(false) << "unsuported texture memory";
    }
  }

  param_storage_size_ = 0;
  buffer_storage_size_ = 0;
  for (size_t sid = 0; sid < pool_entry_.size(); sid++) {
    const auto &pit = pool_entry_[sid];
    // if (pit.params_entry)
    //   continue;
    const auto &cit = std::find_if(devices_.begin(), devices_.end(), [&pit](const DLDevice &d) {
      return pit.device_type == static_cast<int>(d.device_type);
    });
    DLDevice dev = cit == devices_.end() ? devices_[0] : *cit;
    std::vector<int64_t> shape = pit.shape;
    if (shape.size() == 1) {
      shape[0] = (shape[0] + 3) / 4;
    }
    ::tvm::runtime::Optional<::tvm::runtime::String> mem_scope;
    if (!pit.scope.empty()) {
      mem_scope = ::tvm::runtime::String(pit.scope);
    }
    CHECK(dev.device_type == kDLCUDA && dev.device_id == 0);
    sta::STensor last_tensor;
    if (!alloc) {
      storage_pool_.push_back(sta::Null(shape, pit.dtype));
    } else {
      storage_pool_.push_back(sta::Empty(shape, at::MemoryFormat::Contiguous, pit.dtype, sta::MemType::kInfer));
    }
    last_tensor = storage_pool_.back();
    // op_node_storage_id_map_[sid] = storage_pool_.size() - 1;
    if (pit.params_entry) {
      param_storage_size_ += ::tvm::runtime::GetDataSize(*last_tensor.operator->());
    } else {
      buffer_storage_size_ += ::tvm::runtime::GetDataSize(*last_tensor.operator->());
    }
  }

  data_entry_.resize(tvm_graph_.node_row_ptr_.back());
  data_alignment_.resize(tvm_graph_.node_row_ptr_.back());
  for (size_t i = 0; i < data_entry_.size(); i++) {
    int storage_id = tvm_graph_.attrs_.storage_id[i];
    
    data_entry_[i] = sta::ViewShapeDtype(storage_pool_[storage_id], tvm_graph_.attrs_.shape[i], vtype[i]);
    data_alignment_[i] = details::GetDataAlignment(*data_entry_[i].operator->());
  }

  // setup cpu pin memory
  for (auto nid : tvm_graph_.input_nodes_) {
    if (!tvm_graph_.params_.count(nid)) {
      auto & input_id = tvm_graph_.nodes_[nid].name;
      auto & shape = tvm_graph_.attrs_.shape[nid];
      auto & dtype = tvm_graph_.attrs_.dltype[nid];
      input_cpu_pin_bufs_[input_id] = ::tvm::runtime::NDArray::Empty(
          shape, ::tvm::runtime::String2DLDataType(dtype), {kDLCUDAHost, 0});
    }
  }
  for (auto e : tvm_graph_.outputs_) {
    auto nid = entry_id(e);
    auto & output_id = tvm_graph_.nodes_[nid].name;
    auto & shape = tvm_graph_.attrs_.shape[nid];
    auto & dtype = tvm_graph_.attrs_.dltype[nid];
    output_cpu_pin_bufs_[output_id] = ::tvm::runtime::NDArray::Empty(
        shape, ::tvm::runtime::String2DLDataType(dtype), {kDLCUDAHost, 0});
  }

  auto model_name_without_dup_id = GetModelNameWithoutDuplicatedId(tvm_graph_.model_name_);

  static std::set<std::string> logged;
  if (!logged.count((model_name_without_dup_id))) {
    logged.insert(model_name_without_dup_id);
    LOG(INFO) << "[Executor] " << model_name_without_dup_id
              << " params " << 1.0 * param_storage_size_ / 1024 / 1024 << " Mb"
              << " intermediate " << 1.0 * buffer_storage_size_ / 1024 / 1024 << " Mb";
  }
}

void Executor::SetupOpExecs() {
  op_execs_.resize(tvm_graph_.nodes_.size());
  input_dltensors_.resize(tvm_graph_.node_row_ptr_.back());
  output_dltensors_.resize(tvm_graph_.node_row_ptr_.back());
  both_input_output_dltensors_.resize(tvm_graph_.node_row_ptr_.back());
  input_param_nid_.resize(op_execs_.size());
  std::unordered_set<uint32_t> input_node_eids;
  for (size_t i = 0; i < tvm_graph_.input_nodes_.size(); i++) {
    uint32_t nid = tvm_graph_.input_nodes_[i];
    input_node_eids.insert(tvm_graph_.node_row_ptr_[nid]);
  }
  std::unordered_set<uint32_t> output_node_eids;
  for (uint32_t i = 0; i < tvm_graph_.outputs_.size(); i++) {
    auto& output = tvm_graph_.outputs_[i];
    output_node_eids.insert(tvm_graph_.node_row_ptr_[output.node_id] + output.index);
  }
  
  for (uint32_t nid = 0; nid < GetNumOfNodes(); nid++) {
    const auto &inode = tvm_graph_.nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor*> args;
    for (const auto& e : inode.inputs) {
      uint32_t eid = entry_id(e);
      // args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
      sta::STensor data_entry_view_tensor = data_entry_[eid];
      args.push_back(data_entry_view_tensor.MutableDLTensor());

      if (tvm_graph_.params_.count(e.node_id)) {
        input_param_nid_[nid].push_back(e.node_id);
      }
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; index++) {
      uint32_t eid = entry_id(nid, index);
      // args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
      sta::STensor data_entry_view_tensor = data_entry_[eid];
      args.push_back(data_entry_view_tensor.MutableDLTensor());
    }
    CHECK(inode.op_type == "tvm_op");

    std::shared_ptr<OpArgs> op_args = nullptr;
    std::tie(op_execs_[nid], op_args) = CreateTVMOp(inode.param, args);

    for (size_t i = 0; i < inode.inputs.size(); i++) {
      uint32_t input_eid = entry_id(inode.inputs[i]);
      if (input_node_eids.count(input_eid)) {
        input_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
      if (output_node_eids.count(input_eid)) {
        both_input_output_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }

    for (uint32_t i = inode.inputs.size(); i < inode.inputs.size() + inode.param.num_outputs; i++) {
      uint32_t output_eid = this->entry_id(nid, i - inode.inputs.size());
      if (output_node_eids.count(output_eid)) {
        output_dltensors_[output_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }
  }
}

std::pair<std::function<void()>, std::shared_ptr<OpArgs>> Executor::CreateTVMOp(
    const TVMOpParam &param, const std::vector<DLTensor*> &args) {
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  arg_ptr->args = args;
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); i++) {
    TVMValue v;
    v.v_handle = static_cast<void*>(args[i]);
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kTVMDLTensorHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] =
          std::accumulate(args[i]->shape, args[i]->shape + args[i]->ndim, 1, std::multiplies<int64_t>());
      args[i]->ndim = 1;
      args[i]->shape = &(arg_ptr->shape_data[i]);
    }
  }

  if (param.func_name == "__nop") {
    return {[]() {}, arg_ptr};
  } else if (param.func_name == "__copy") {
    auto fexec = [arg_ptr]() {
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      TVM_CCALL(TVMArrayCopyFromTo(from, to, nullptr));
    };
    return {fexec, arg_ptr};
  }
  ::tvm::runtime::PackedFunc pf = tvm_graph_.module_.GetFunction(param.func_name, true);
  CHECK(pf != nullptr);
  
  auto fexec = [arg_ptr, pf]() {
    ::tvm::runtime::TVMRetValue rv;
    ::tvm::runtime::TVMArgs targs(arg_ptr->arg_values.data(), arg_ptr->arg_tcodes.data(),
                                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return {fexec, arg_ptr};
}

void Executor::AllocStorageMaybeAdjust() {
  // CHECK(Config::use_shared_tensor_infer && Config::ondemand_adjust);
  CHECK(Config::ondemand_adjust);
  CHECK(!Config::infer_raw_blob_alloc);

  bool adjusted = false;

  auto adjust_train_batch_size = [this, &adjusted](bool first_adjust) {
    if (adjusted) return;
    if (!Controller::Get()->IsTrainIdle()) {
      auto wait_train_pid = TrainLauncher::Get()->GetTrainPid();
      // LOG(INFO) << "[Executor] AllocStorageMaybeAdjust: model " << this->tvm_graph_.model_rank_ 
      //           << " begin wait train pid " << wait_train_pid;
      // this->tvm_graph_.infer_model_->SetWaitTrainPid(this->infer_model_worker_id_, wait_train_pid);

      PROFILE_START(TrainAdjust);
      auto adjust_batch_size = TrainLauncher::Get()->
          GetAdjustBatchSize(sta::ByteToMB(GetStorageSizeAlign()));
      auto cmd_id = Controller::Get()->
          ColocateAdjust(this->tvm_graph_.model_rank_, adjust_batch_size);
      Controller::Get()->WaitColocateAdjustDone(cmd_id);
      PROFILE_END(TrainAdjust);
      if (first_adjust) {
        Profiler::Get()->RecordPerf(Profiler::PerfItem::TrainFirstAdjust, PROFILE_DURATRION(TrainAdjust));        
      }
      LOG(INFO) << "[Executor] AllocStorageMaybeAdjust: model " << this->tvm_graph_.model_rank_ 
                << " wait adjust " << PROFILE_DURATRION(TrainAdjust)
                << " wait train pid " << wait_train_pid;
                
    } else {
      LOG(INFO) << "[Executor] AllocStorageMaybeAdjust: model "
                << this->tvm_graph_.model_rank_ << " train idle";
    }
    adjusted = true;
  };

  // ensure sequential inference allocation  
  // if (!Controller::Get()->TryEnterInferChangeMemory(tvm_graph_.model_rank_)) {
  if (!ResourceManager::InferChangeMemoryTryLock()) {
    if (Controller::Get()->HasFlyingColocateAdjust()) {
      adjust_train_batch_size(false);
    }
    auto _t0 = Profiler::Now();
    // Controller::Get()->EnterInferChangeMemory(tvm_graph_.model_rank_);
    ResourceManager::InferChangeMemoryLock();
    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferWaitBeforeEnterAlloc, Profiler::MilliFrom(_t0));
  }

  double free_memory_mb = ResourceManager::GetFreeMemoryMB();
  // double free_memory_mb = GetFreeMemoryMB();
  // if (!Controller::Get()->IsTrainIdle()) {
  //   if (Config::use_shared_tensor_train) {
  //     free_memory_mb = sta::ByteToMB(sta::CUDAMemPool::PoolNbytes() - sta::CUDAMemPool::InferMemUsage());
  //     free_memory_mb -= std::max(sta::ByteToMB(sta::CUDAMemPool::TrainAllMemUsage()),
  //                                TrainLauncher::Get()->PredictMemUsageMB());
  //     free_memory_mb -= Config::train_memory_over_predict_mb;
  //   } else {
  //     auto [free, total] = Profiler::GetGPUMemInfo();
  //     auto infer_memory_mb = sta::ByteToMB(Profiler::GetLastInferMem());
  //     auto train_memory_mb = std::max(sta::ByteToMB(Profiler::GetLastTrainMem()),
  //                                           TrainLauncher::Get()->PredictMemUsageMB());
  //     train_memory_mb += Config::train_memory_over_predict_mb;
  //     free_memory_mb = std::min(sta::ByteToMB(free), 
  //                               sta::ByteToMB(total) - infer_memory_mb - train_memory_mb);
  //     // free_memory_mb -= Config::train_memory_over_predict_mb;
  //     LOG(INFO) << "free " << sta::ByteToMB(free) << " total " << sta::ByteToMB(total)
  //               << " infer memory " << infer_memory_mb << " train memory " << train_memory_mb 
  //               << " predict train memory " << TrainLauncher::Get()->PredictMemUsageMB()
  //               << " free memory " << free_memory_mb;
  //   }
  // }

  size_t total_storage_nbytes = 0;
  std::vector<size_t> storage_nbytes(storage_pool_.size());
  for (size_t sid = 0; sid < storage_pool_.size(); sid++) {
    auto tensor = storage_pool_[sid];
    auto nbytes = sta::ComputeStorageNbytes(tensor.Shape(), tensor.Stride(), 
        tensor->dtype, tensor.StorageOffset());
    storage_nbytes[sid] = nbytes;
    total_storage_nbytes += sta::detail::GetAlignedNbytes(nbytes);
  }

  LOG(INFO) << "infer require " << sta::ByteDisplay(total_storage_nbytes)
            << " free memory " << free_memory_mb << " MB";
  if (sta::ByteToMB(total_storage_nbytes) > free_memory_mb) {
    adjust_train_batch_size(true);
  }

  PROFILE_START(InferAllocStorage);
  AllocStorage();
  PROFILE_END(InferAllocStorage);

  // TODO: consider fwd/bwd
  // if (sta::ByteToMB(total_storage_nbytes) < free_memory_mb) {
  // // case 1: memory is enough, do not need to adjust
  //   for (size_t sid = 0; sid < storage_pool_.size(); sid++) {
  //     auto &s = storage_pool_[sid];
  //     auto tensor = sta::TensorPool::Get()->Tensor(s);
  //     tensor.AllocForNull(sta::MemType::kInfer, false);
  //   }
  // } else { 
  // // casae 2: memory is not enough, adjustment batch
  //   for (size_t sid = 0; sid < storage_pool_.size(); ) {
  //     auto &s = storage_pool_[sid];
  //     auto tensor = sta::TensorPool::Get()->Tensor(s);
  //     auto nbytes = sta::ComputeStorageNbytes(tensor.Shape(), tensor.Stride(), 
  //         tensor->dtype, tensor.StorageOffset());
  //     auto mdata = sta::CUDAMemPool::Get()->Alloc(nbytes, sta::MemType::kInfer, true);
  //     if (nbytes > 0 && mdata == nullptr) {
  //       adjust_train_batch_size();
  //     } else {
  //       tensor.SetMDataForNull(mdata);
  //       sid++;
  //     }
  //   }
  // }
  // Controller::Get()->ExitInferChangeMemory(tvm_graph_.model_rank_);
  ResourceManager::InferChangeMemoryUnlock();
}

// double Executor::GetFreeMemoryMB() {
//   double free_memory_mb;
//   if (!Controller::Get()->IsTrainIdle()) {
//     if (Config::use_shared_tensor_train) {
//       free_memory_mb = sta::ByteToMB(sta::CUDAMemPool::PoolNbytes() -
//                                              sta::CUDAMemPool::InferMemUsage());
//       free_memory_mb -= TrainLauncher::Get()->PredictMemUsageMB();
//           // std::max(sta::ByteToMB(sta::CUDAMemPool::TrainAllMemUsage()),
//           //          TrainLauncher::Get()->PredictMemUsageMB());
//       free_memory_mb -= Config::train_memory_over_predict_mb;
//     } else {
//       auto [free, total] = Profiler::GetGPUMemInfo();
//       auto infer_memory_mb = sta::ByteToMB(Profiler::GetLastInferMem());
//       auto train_memory_mb = TrainLauncher::Get()->PredictMemUsageMB();
//           // std::max(sta::ByteToMB(Profiler::GetLastTrainMem()),
//           //          TrainLauncher::Get()->PredictMemUsageMB());
//       train_memory_mb += Config::train_memory_over_predict_mb;
//       free_memory_mb = std::min(
//           sta::ByteToMB(free),
//           sta::ByteToMB(total) - infer_memory_mb - train_memory_mb);
//       // free_memory_mb -= Config::train_memory_over_predict_mb;
//       LOG(INFO) << "free " << sta::ByteToMB(free) << " total "
//                 << sta::ByteToMB(total) << " infer memory "
//                 << infer_memory_mb << " train memory " << train_memory_mb
//                 << " predict train memory "
//                 << TrainLauncher::Get()->PredictMemUsageMB()
//                 << " free memory " << free_memory_mb;
//     }
//   }
//   return free_memory_mb;
// }

}  // namespace tvm
}