#include <server/logging_as_glog.h>
#include <server/train_launcher.h>
#include <server/model_store/infer_model_store.h>
#include <server/profiler.h>
#include <server/control/controller.h>
#include <server/resource_manager.h>
#include <server/config.h>
#include <server/tvm/executor.h>
#include <server/tvm/graph.h>

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>

#include <common/tensor/shape_helper.h>
#include <common/tensor/dtype_helper.h>
#include <common/util.h>

#include <boost/range/irange.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <regex>
#include <utility>


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


Executor::Executor(TVMGraph &tvm_graph, size_t worker_id, 
                   const std::vector<DLDevice> &devs)
    : tvm_graph_(tvm_graph), infer_model_worker_id_(worker_id), devices_(devs), 
      initialized_(false), cold_cached_nbytes_(0) {
  // currently, we still assume that a model is only used in one device
  // we will support multi-device inference in the future
  
  using namespace ::tvm::runtime;
  SetupStorage(false);
  SetupOpExecs();

  exec_stream_ = DeviceAPI::Get(devices_[0])
      ->CreateStream(devices_[0]);
  load_param_stream_ = DeviceAPI::Get(devices_[0])
      ->CreateStream(devices_[0]);

  param_ready_.resize(tvm_graph_.node_row_ptr_.back());
  for (size_t i = 0; i < param_ready_.size(); i++) {
    param_ready_[i] = std::make_unique<std::atomic<bool>>(false);
  }
  param_ready_events_.resize(tvm_graph_.node_row_ptr_.back());
  param_ready_event_ids_.resize(tvm_graph_.node_row_ptr_.back(), 
                                static_cast<uint32_t>(-1));
  for (size_t i = 0; i < param_ready_events_.size(); i++) {
    COL_CUDA_CALL(cudaEventCreateWithFlags(&param_ready_events_[i], 
                                           cudaEventDisableTiming));
  }

  if (Config::group_param_load) {
    auto num_param_groups = tvm_graph_.host_param_group_.size();
    for (auto pg_id : boost::irange(num_param_groups)) {
      auto & pg = tvm_graph_.host_param_group_[pg_id];
      for (auto param_eid : pg.second) {
        param_ready_event_ids_[param_eid] = pg_id;
      }
    }
  } else {
    for (auto &p : tvm_graph_.host_params_) {
      param_ready_event_ids_[p.first] = p.first;
    }
  }
}

void Executor::Init(bool load_param) {
  if (!initialized_) {   
    PROFILE_START(InferAllocStorage);
    if (!Config::colocate_config.skip_malloc) AllocStorage();
    PROFILE_END(InferAllocStorage);
     
    if (!Config::colocate_config.skip_malloc)
      RefreshDataEntry();

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
    RefreshDataEntry();
  }
  if (load_param) {
    LOG(INFO) << "FakeInit load_param, skip load_param in Init";
    LoadParams(false, true);
  }
}

void Executor::DeInit(const std::vector<size_t> &keep_cold_cached_group_id) {
  CHECK(initialized_);
  cold_cached_group_.clear();
  size_t cold_cached_nbytes = 0;
  for (size_t k : keep_cold_cached_group_id) {
    size_t aligned_nbytes = tvm::GetMemBlockAlignedNBytes(storage_group_.at(k)->nbytes);
    CHECK_EQ(cold_cached_group_.emplace(
        std::make_pair(k, storage_group_.at(k))).second, true);
    cold_cached_nbytes += aligned_nbytes;
    DLOG(INFO) << "Keep " << tvm_graph_.model_name_ 
              << "cold cached group " << k << " nbytes " 
              << sta::PrintByte(aligned_nbytes);
  }

  LOG_IF(INFO, Config::log_infer_model_reclaim) 
      << "[Executor] " << tvm_graph_.model_name_ 
      << " deinit, cold_cached_nbytes = " 
      << sta::PrintByte(cold_cached_nbytes) << ".";
  cold_cached_nbytes_.store(cold_cached_nbytes, std::memory_order_relaxed);
  if (!Config::colocate_config.skip_malloc) {
    ResetStorage();
  }
  initialized_ = false;
}

void Executor::ClearColdCached(const std::vector<size_t> &cold_cached_group_id) {
  CHECK(!initialized_);
  CHECK_EQ(cold_cached_group_.size(), cold_cached_group_id.size());
  for (size_t group_id : cold_cached_group_id) {
    CHECK_EQ(cold_cached_group_.count(group_id), 1);
  }
  cold_cached_group_.clear();
  cold_cached_nbytes_.store(0, std::memory_order_relaxed);
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
      for (auto eid : input_param_eid_[i]) {
        std::unique_lock pipeline_load_param_lock{pipeline_load_params_mut_};
        pipeline_load_params_cv_.wait(pipeline_load_param_lock, [this, eid] {
          return param_ready_[eid]->load();
        });
      }
      wait_load_ms += Profiler::MilliFrom(t0);

      auto t1 = Profiler::Now();
      // 2. wait load finish
      for (auto eid : input_param_eid_[i]) {
        auto event_id = param_ready_event_ids_[eid];
        CHECK(event_id != -1);
        COL_CUDA_CALL(cudaStreamWaitEvent((cudaStream_t)exec_stream_, 
                      param_ready_events_[event_id]));
      }
      wait_ms += Profiler::MilliFrom(t1);

      op_execs_[i]();
    }
  }
  DeviceAPI::Get(devices_[0])->StreamSync(devices_[0], exec_stream_);
  LOG_IF(INFO, Config::log_infer_pipeline_exec) 
      << "[Executor] [PipelineRun] " << tvm_graph_.model_name_
      << " wait_load_ms " << wait_load_ms 
      << " wait_ms " << wait_ms
      << " tot " << Profiler::MilliFrom(begin);
}

const DLTensor* Executor::GetInput(int index) const {
  CHECK(initialized_);
  CHECK_LT(static_cast<size_t>(index), tvm_graph_.input_nodes_.size());
  uint32_t eid = tvm_graph_.entry_id(tvm_graph_.input_nodes_[index], 0);
  return data_entry_[eid].operator->();
}

const DLTensor* Executor::GetInput(const std::string &index) const {
  return GetInput(GetInputIndex(index));
}

const DLTensor* Executor::GetInputHostBuf(const std::string &index) const {
  CHECK(initialized_);
  return input_host_pin_bufs_.at(index).operator->();
}

const DLTensor* Executor::GetOutput(int index) const {
  CHECK(initialized_);
  CHECK_LE(static_cast<size_t>(index), tvm_graph_.outputs_.size());
  uint32_t eid = tvm_graph_.entry_id(tvm_graph_.outputs_[index]);
  return data_entry_[eid].operator->();
}

const DLTensor* Executor::GetOutput(const std::string &index) const {
  return GetOutput(GetOutputIndex(index));
}

const DLTensor* Executor::GetOutputHostBuf(const std::string &index) const {
  CHECK(initialized_);
  return output_host_pin_bufs_.at(index).operator->();
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

void Executor::ResetStorage() {
  using namespace ::tvm::runtime;
  for (auto &p : tvm_graph_.host_params_) {
    param_ready_[p.first]->store(false);
  }
  for (auto & e : data_entry_) { e.DeallocToNull(); }
  for (auto & s : storage_pool_) { s.DeallocToNull(); }
  if (Config::use_shared_tensor_infer) {
    if (Config::better_alloc) { storage_group_.clear(); }
  } else {
    if (Config::infer_raw_blob_alloc) { blob_mem_.reset(); }
  }
}

void Executor::AllocStorage() {
  using namespace ::tvm::runtime;

  if (Config::use_shared_tensor_infer) {
    if (!Config::better_alloc) {
      for (auto &s : storage_pool_) {
        s.AllocForNull(sta::MemType::kInfer);
      }
    } else {
      for (auto k : boost::irange(tvm_graph_.storage_group_nbytes_.size())) {
        auto group_nbytes = tvm_graph_.storage_group_nbytes_[k];
        std::shared_ptr<sta::CUDAMemPool::PoolEntry> mdata_group;
        if (auto it = cold_cached_group_.find(k); it != cold_cached_group_.end()) {
          mdata_group = it->second;
        } else {
          mdata_group = sta::CUDAMemPool::Get(devices_[0].device_id)->Alloc(
              group_nbytes, sta::MemType::kInfer, false);
        }
        storage_group_.push_back(mdata_group);

        // create storage pool
        size_t check_off = 0;
        auto group_left = tvm_graph_.storage_group_partition_[k];
        auto group_right = tvm_graph_.storage_group_partition_[k + 1];
        for (auto i : boost::irange(group_left, group_right)) {
          auto offset = tvm_graph_.storage_group_offsets_[k][i - group_left];
          auto tensor = storage_pool_[tvm_graph_.storage_alloc_order_[i]];
          CHECK(tensor.IsNull());
          CHECK_EQ(offset,  check_off);

          auto aligned_nbytes = GetLineAlignedNbytes(tensor.ComputeNbytes());
          auto mdata = std::shared_ptr<sta::CUDAMemPool::PoolEntry>(
              new sta::CUDAMemPool::PoolEntry{
                static_cast<char*>(mdata_group->addr) + offset, 
                aligned_nbytes});
          tensor.SetMDataForNull(mdata);
          check_off += aligned_nbytes;
        }
      }
    }
  } else if (Config::infer_raw_blob_alloc) {
    // deprecated
    CHECK(false) << "infer_raw_blob_alloc is deprecated";
#if 0
    size_t total_nbytes = 0, off = 0;
    // constexpr size_t align = 4 * sizeof(int);
    constexpr size_t align = 1;
    static_assert(((align - 1) & align) == 0, "align must be power of 2");
    for (auto &s : storage_pool_) {
      total_nbytes += (GetDataSize(*s.operator->()) + align - 1) & (~(align - 1));
    }
    blob_mem_ = sta::CUDAMemPool::Get(devices_[0].device_id)->RawAlloc(
        total_nbytes, sta::MemType::kInfer);
    // blob_mem_ = sta::CUDAMemPool::Get()->Alloc(total_nbytes, sta::MemType::kInfer);
    for (auto &s : storage_pool_) {
      size_t nbytes = (GetDataSize(*s.operator->()) + align - 1) & (~(align - 1));
      auto mdata = std::shared_ptr<sta::CUDAMemPool::PoolEntry>(
          new sta::CUDAMemPool::PoolEntry{static_cast<char*>(blob_mem_->addr) + off, nbytes});
      s.SetMDataForNull(mdata);
      off += nbytes;
    }
#endif
  } else {
    for (auto &s: storage_pool_) {
      s.AllocForNull(sta::MemType::kInfer);
    }
  }
}

void Executor::LoadParams(bool pipeline, bool force) {
  using namespace ::tvm::runtime;
  if (force) {
    for (auto &p : tvm_graph_.host_params_)
      param_ready_[p.first]->store(false);
  }
  bool cold_cache_hit = false;
  auto t_a = Profiler::Now();

  auto load_param_t = Profiler::Now();
  auto call_api_t = Profiler::Now();

  if (!Config::group_param_load) {
    for (auto &p : tvm_graph_.host_params_) {
      auto sid = tvm_graph_.attrs_.storage_id[p.first];
      if (!param_ready_[p.first]->load()) {
        sta::STensor storage_tensor = storage_pool_[sid];
        tvm::TVMArray::CopyFromTo(p.second.operator->(), 
                                  storage_tensor.MutableDLTensor(), 
                                  load_param_stream_);
        if (pipeline) {
          param_ready_[p.first]->store(true);
          pipeline_load_params_cv_.notify_all();
          CHECK(param_ready_event_ids_[p.first] == p.first);
          COL_CUDA_CALL(cudaEventRecord(param_ready_events_[p.first], 
                                        (cudaStream_t)load_param_stream_));
        }
      }
    }
  } else {
    size_t sg_id = 0; /*  next storage group index */
    size_t sg_off = 0;
    auto storage_group = storage_group_[sg_id++];
    cold_cache_hit = !cold_cached_group_.empty();
    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferModelColdCacheHit, 
                                cold_cache_hit);

    int num_param_groups = tvm_graph_.host_param_group_.size();
    for (auto pg_id : boost::irange(num_param_groups)) {
      auto & pg = tvm_graph_.host_param_group_[pg_id];
      auto & param_group = pg.first;
      auto param_group_nbytes = static_cast<size_t>(param_group->shape[0]);
      auto & param_ids = pg.second;
      for (size_t pg_off = 0; pg_off < param_group_nbytes; ) {
        auto load_nbytes = std::min(param_group_nbytes - pg_off, 
                                    storage_group->nbytes - sg_off);
        COL_CUDA_CALL(cudaSetDevice(devices_[0].device_id));
        if (auto sg_it = cold_cached_group_.find(sg_id - 1); 
            sg_it == cold_cached_group_.cend()) {
          COL_CUDA_CALL(cudaMemcpyAsync(
              static_cast<char*>(storage_group->addr) + sg_off,
              static_cast<char*>(param_group->data) + pg_off,
              load_nbytes, cudaMemcpyDefault, 
              (cudaStream_t)load_param_stream_));
          std::stringstream ss;
          for (auto &&[id, arr] : cold_cached_group_) {
            ss << id  << "," << arr->nbytes << "|";
          }
          for (auto nbytes : tvm_graph_.storage_group_nbytes_) {
            ss << nbytes << "@";
          }
          if (Config::cold_cache_ratio == 1.0) {
            CHECK(cold_cached_group_.empty()) 
                << sg_id << " not cached, cold_cached_group = " 
                << ss.str() << ", storage group size" 
                << storage_group_.size() << ".";
          }
        } else {
          CHECK_LE(sg_id, cold_cached_group_.size());
          CHECK_EQ(sg_it->second, storage_group)
              << "cached storage group not match, cached addr "
              << sg_it->second->addr << " required addr "
              << storage_group->addr;
        }
        sg_off += load_nbytes;
        pg_off += load_nbytes;
        if (sg_off == storage_group->nbytes) {
          CHECK_LT(sg_id, storage_group_.size());
          storage_group = storage_group_[sg_id++];
          sg_off = 0;
        }
      }
      // because we iterate param group in out loop, 
      // we only need to record after param group transited
      CHECK_EQ(param_ready_event_ids_[param_ids[0]], pg_id);
      COL_CUDA_CALL(cudaEventRecord(param_ready_events_[pg_id], 
                    (cudaStream_t)load_param_stream_));
      if (pipeline) {
        for (auto & pid : param_ids) {
          param_ready_[pid]->store(true);
        }
        pipeline_load_params_cv_.notify_all();
      }
    }
    // auto tot_ms = Profiler::MilliFrom(t_b);
    // LOG(INFO) << "Load Params" << " api_1_ms " << api_1_ms << " api_2_ms " << api_2_ms << " tot_ms " << tot_ms;

  }

  auto call_api_ms = Profiler::MilliFrom(call_api_t);

  // because we load params in async thread in pipeline mode,
  // it's ok to sync to calculate loading time
  ::tvm::runtime::DeviceAPI::Get(devices_[0])
      ->StreamSync(devices_[0], load_param_stream_);
  if (!pipeline) {
    for (auto &p : tvm_graph_.host_params_)
      param_ready_[p.first]->store(true);
  }

  LOG_IF(INFO, Config::log_infer_load_param) 
      << "[Executor] [LoadParamas] "
      << tvm_graph_.model_name_
      << " call_api_ms " << call_api_ms
      << " tot_ms " << Profiler::MilliFrom(load_param_t)
      << " cold_cache_hit " << cold_cache_hit
      << " total_nbytes " << tvm_graph_.GetParamStorageNbytes()
      << " cached_nbytes " << cold_cached_nbytes_.load(std::memory_order_relaxed);
}

void Executor::RefreshDataEntry() {
  for (size_t i = 0; i < data_entry_.size(); i++) {
    int sid = tvm_graph_.attrs_.storage_id[i];
    auto storage_tensor = storage_pool_[sid];
    data_entry_[i].SetMDataForNull(storage_tensor.MData());
  }
}


void Executor::SetupStorage(bool alloc) {
  std::vector<DLDataType> vtype;
  for (const std::string &s_type : tvm_graph_.attrs_.dltype) {
    vtype.push_back(::tvm::runtime::String2DLDataType(s_type));
  }
  
  for (auto & entry : tvm_graph_.storage_pool_entries_) {
    storage_pool_.push_back(sta::Null({static_cast<int64_t>(entry.nbytes)}, 
        devices_[0], sta::DLInt8));
  }

  data_entry_.resize(tvm_graph_.node_row_ptr_.back());
  for (size_t i = 0; i < data_entry_.size(); i++) {
    int storage_id = tvm_graph_.attrs_.storage_id[i];
    
    data_entry_[i] = sta::ViewShapeDtype(storage_pool_[storage_id], 
                                         tvm_graph_.attrs_.shape[i], vtype[i]);
  }

  SetupHostPinnedIOStorage();

  if (alloc) {
    AllocStorage();
    RefreshDataEntry();
  }

}

void Executor::SetupHostPinnedIOStorage() {
  // setup cpu pin memory
  for (auto nid : tvm_graph_.input_nodes_) {
    auto eid = tvm_graph_.entry_id(nid, 0);
    if (!tvm_graph_.host_params_.count(eid)) {
      auto & input_id = tvm_graph_.nodes_[nid].name;
      auto & shape = tvm_graph_.attrs_.shape[eid];
      auto & dtype = tvm_graph_.attrs_.dltype[eid];
      input_host_pin_bufs_[input_id] = ::tvm::runtime::NDArray::Empty(
          shape, ::tvm::runtime::String2DLDataType(dtype), 
          {kDLCUDAHost, 0});
    }
  }
  for (auto e : tvm_graph_.outputs_) {
    auto nid = e.node_id;
    auto eid = tvm_graph_.entry_id(e);
    auto & output_id = tvm_graph_.nodes_[nid].name;
    auto & shape = tvm_graph_.attrs_.shape[eid];
    auto & dtype = tvm_graph_.attrs_.dltype[eid];
    output_host_pin_bufs_[output_id] = ::tvm::runtime::NDArray::Empty(
        shape, ::tvm::runtime::String2DLDataType(dtype), 
        {kDLCUDAHost, 0});
  }
}

void Executor::SetupOpExecs() {
  op_execs_.resize(tvm_graph_.nodes_.size());
  input_param_eid_.resize(op_execs_.size());

  std::unordered_set<uint32_t> input_node_eids;
  std::unordered_set<uint32_t> output_node_eids;

  for (size_t i = 0; i < tvm_graph_.input_nodes_.size(); i++) {
    uint32_t nid = tvm_graph_.input_nodes_[i];
    input_node_eids.insert(tvm_graph_.node_row_ptr_[nid]);
  }
  
  for (uint32_t i = 0; i < tvm_graph_.outputs_.size(); i++) {
    auto& output = tvm_graph_.outputs_[i];
    output_node_eids.insert(
        tvm_graph_.node_row_ptr_[output.node_id] + output.index);
  }
  
  for (uint32_t nid = 0; nid < GetNumOfNodes(); nid++) {
    const auto &inode = tvm_graph_.nodes_[nid];
    if (inode.op_type == "null") continue;

    std::vector<DLTensor*> args;
    for (const auto& e : inode.inputs) {
      uint32_t eid = tvm_graph_.entry_id(e);
      sta::STensor data_entry_view_tensor = data_entry_[eid];
      args.push_back(data_entry_view_tensor.MutableDLTensor());

      if (tvm_graph_.host_params_.count(eid)) {
        input_param_eid_[nid].push_back(eid);
      }
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; index++) {
      uint32_t eid = tvm_graph_.entry_id(nid, index);
      sta::STensor data_entry_view_tensor = data_entry_[eid];
      args.push_back(data_entry_view_tensor.MutableDLTensor());
    }
    CHECK(inode.op_type == "tvm_op");

    std::shared_ptr<OpArgs> op_args = nullptr;
    std::tie(op_execs_[nid], op_args) = CreateTVMOp(inode.param, args);
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
      arg_ptr->shape_data[i] = std::accumulate(args[i]->shape, 
                                               args[i]->shape + args[i]->ndim, 
                                               1, std::multiplies<int64_t>());
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
  ::tvm::runtime::PackedFunc pf = 
      tvm_graph_.module_.GetFunction(param.func_name, true);
  CHECK(pf != nullptr);
  
  auto fexec = [arg_ptr, pf]() {
    ::tvm::runtime::TVMRetValue rv;
    ::tvm::runtime::TVMArgs targs(arg_ptr->arg_values.data(), 
                                  arg_ptr->arg_tcodes.data(),
                                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return {fexec, arg_ptr};
}

size_t Executor::GetMissingStorageSizeAlign() const {
  CHECK_GE(tvm_graph_.GetStorageAlignedNbytes(),
           cold_cached_nbytes_.load(std::memory_order_relaxed));
  return tvm_graph_.GetStorageAlignedNbytes() -
         cold_cached_nbytes_.load(std::memory_order_relaxed);
}

} // namespace tvm
} // namespace colserve