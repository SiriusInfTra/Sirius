#include <numeric>
#include <thread>
#include <c10/core/MemoryFormat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "graph_executor.h"
#include <sta/shape_helper.h>

#include "../model_train_store.h"
#include "../model_infer_store.h"
#include "../profiler.h"
#include "../controller.h"
#include "../config.h"


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

#define CU_CALL(func) \
  do { \
    auto err = func; \
    if (err != CUDA_SUCCESS) { \
      const char* pstr = nullptr; \
      cuGetErrorString(err, &pstr); \
      LOG(FATAL) << #func << ": " << pstr; \
    } \
  } while (0);

GraphExecutor::GraphExecutor(GraphExecutorFactory &factory, size_t worker_id)
    : factory_(factory), infer_model_worker_id_(worker_id), initialized_(false) {
  using namespace ::tvm::runtime;
  auto t0 = std::chrono::steady_clock::now();
  SetupStorage(false);
  SetupOpExecs();
  exec_stream_ = DeviceAPI::Get(factory_.devices_[0])
      ->CreateStream(factory_.devices_[0]);
  load_param_stream_ = DeviceAPI::Get(factory_.devices_[0])
      ->CreateStream(factory_.devices_[0]);

  param_ready_.resize(GetNumOfNodes(), false);
  auto t1 = std::chrono::steady_clock::now();
  DLOG(INFO) << "[GraphExecutor] Create " 
             << std::chrono::duration<double, std::milli>(t1-t0).count() << "ms";
}

void GraphExecutor::Init() {
  if (!initialized_) {   
    if (!Config::ondemand_adjust) {
      PROFILE_START(InferAllocStorage, 0);
      if (!Config::colocate_config.skip_malloc) AllocStorage();
      PROFILE_END(InferAllocStorage, 0);
    } else {
      PROFILE_START(InferAdjustAlloc, 0);
      AllocStorageMaybeAdjust();
      PROFILE_END(InferAdjustAlloc, 0);
    }
    
    if (!Config::colocate_config.skip_malloc)
      ReSetupDataEntry();

    PROFILE_START(InferLoadParam, 0);
    if (!Config::colocate_config.skip_loading) {
      LoadParams(false, Config::colocate_config.skip_malloc);
    }
    PROFILE_END(InferLoadParam, 0);
    
    initialized_ = true;
  }
}

void GraphExecutor::FakeInit(bool malloc, bool load_param) {
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

void GraphExecutor::DeInit() {
  CHECK(initialized_);
  if (!Config::colocate_config.skip_malloc) {
    ResetStorage();
  }
  initialized_ = false;
}

void GraphExecutor::Run() {
  using namespace ::tvm::runtime;
  CHECK(initialized_);
  DeviceAPI::Get(factory_.devices_[0])->SetStream(factory_.devices_[0], exec_stream_);
  for (size_t i = 0; i < op_execs_.size(); i++) {
    if (op_execs_[i]) op_execs_[i]();
  }
  DeviceAPI::Get(factory_.devices_[0])->StreamSync(factory_.devices_[0], exec_stream_);
}

void GraphExecutor::PipelineRun() {
  using namespace ::tvm::runtime;
  CHECK(initialized_);
  double wait_time = 0;
  DeviceAPI::Get(factory_.devices_[0])->SetStream(factory_.devices_[0], exec_stream_);
  for (size_t i = 0; i < op_execs_.size(); i++) {
    if (op_execs_[i]) {
      auto t0 = std::chrono::steady_clock::now();
      for (bool param_ready = false; !param_ready; ) {
        param_ready = true;
        for (auto nid : input_param_nid_[i]) {
          if (!param_ready_[nid]) {
            param_ready = false;
            break;
          }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
      auto t1 = std::chrono::steady_clock::now();
      wait_time += std::chrono::duration<double, std::milli>(t1-t0).count();
      op_execs_[i]();
      DeviceAPI::Get(factory_.devices_[0])->StreamSync(factory_.devices_[0], exec_stream_);
    }
  }
  LOG(INFO) << "pipeline run wait "<< wait_time << "ms";
}

const DLTensor* GraphExecutor::GetInput(int index) const {
  CHECK(initialized_);
  CHECK_LT(static_cast<size_t>(index), factory_.input_nodes_.size());
  uint32_t eid = entry_id(factory_.input_nodes_[index], 0);
  if (Config::use_shared_tensor_infer) {
    auto input_view_tensor = sta::TensorPool::Get()->Tensor(data_entry_[eid]);
    return input_view_tensor.operator->();
  } else {
    return raw_data_entry_[eid].operator->();
  }
}

const DLTensor* GraphExecutor::GetInput(const std::string &index) const {
  return GetInput(GetInputIndex(index));
}

const DLTensor* GraphExecutor::GetInputHostBuf(const std::string &index) const {
  CHECK(initialized_);
  return input_cpu_pin_bufs_.at(index).operator->();
}

const DLTensor* GraphExecutor::GetOutput(int index) const {
  CHECK(initialized_);
  CHECK_LE(static_cast<size_t>(index), factory_.outputs_.size());
  uint32_t eid = entry_id(factory_.outputs_[index]);
  if (Config::use_shared_tensor_infer) {
    auto output_view_tensor = sta::TensorPool::Get()->Tensor(data_entry_[eid]);
    return output_view_tensor.operator->();
  } else {
    return raw_data_entry_[eid].operator->();
  }
}

const DLTensor* GraphExecutor::GetOutput(const std::string &index) const {
  return GetOutput(GetOutputIndex(index));
}

const DLTensor* GraphExecutor::GetOutputHostBuf(const std::string &index) const {
  CHECK(initialized_);
  return output_cpu_pin_bufs_.at(index).operator->();
}

uint32_t GraphExecutor::GetInputIndex(const std::string &name) const {
  auto it = factory_.input_map_.find(name);
  if (it != factory_.input_map_.end()) {
    return it->second;
  } else {
    LOG(FATAL) << "cannot find " << name << " in input";
    return -1;
  }
} 

uint32_t GraphExecutor::GetOutputIndex(const std::string &name) const {
  auto it = factory_.output_map_.find(name);
  if (it != factory_.output_map_.end()) {
    return it->second;
  } else {
    LOG(FATAL) << "cannot find " << name << " in output";
    return -1;
  }
}

// void GraphExecutor::ResetBufStorage() {
//   using namespace ::tvm::runtime;
//   for (auto &e : data_entry_) {
//     e.get_mutable()->dl_tensor.data = nullptr;
//   }
//   for (auto &s: storage_pool_) {
//     DeviceAPI::Get(s->device)->FreeDataSpace(s->device, s->data);
//     s.get_mutable()->dl_tensor.data = nullptr;
//   }
// }

// void GraphExecutor::AllocBufStorage() {
//   using namespace ::tvm::runtime;
//   for (auto &s : storage_pool_) {
//     s.get_mutable()->dl_tensor.data =
//         DeviceAPI::Get(s->device)->AllocDataSpace(
//           s->device, s->ndim, s->shape, s->dtype);
//   }
// }

void GraphExecutor::ResetStorage() {
  using namespace ::tvm::runtime;
  for (auto &p : factory_.params_) {
    param_ready_[p.first] = false;
  }
  if (Config::use_shared_tensor_infer) {
    for (auto &e : data_entry_) {
      auto tensor = sta::TensorPool::Get()->Tensor(e);
      tensor.DeallocToNull();
    }
    for (auto &s : storage_pool_) {
      auto tensor = sta::TensorPool::Get()->Tensor(s);
      tensor.DeallocToNull();
    }
  } else {
    for (auto & e : raw_data_entry_) {
      e.DeallocToNull();
    }
    for (auto & s : raw_storage_pool_) {
      s.DeallocToNull();
    }
    if (Config::infer_raw_blob_alloc) {
      blob_mem_.reset();
    }
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

void GraphExecutor::AllocStorage() {
  using namespace ::tvm::runtime;

  if (Config::use_shared_tensor_infer) {
    for (auto &s : storage_pool_) {
      auto tensor = sta::TensorPool::Get()->Tensor(s);
      tensor.AllocForNull(sta::MemType::kInfer, false);
    }
  } else if (Config::infer_raw_blob_alloc) {
    size_t total_nbytes = 0, off = 0;
    // constexpr size_t align = 4 * sizeof(int);
    constexpr size_t align = 1;
    static_assert(((align - 1) & align) == 0, "align must be power of 2");
    for (auto &s : raw_storage_pool_) {
      total_nbytes += (GetDataSize(*s.operator->()) + align - 1) & (~(align - 1));
    }
    blob_mem_ = sta::CUDAMemPool::RawAlloc(total_nbytes, sta::MemType::kInfer);
    // blob_mem_ = sta::CUDAMemPool::Get()->Alloc(total_nbytes, sta::MemType::kInfer);
    for (auto &s : raw_storage_pool_) {
      size_t nbytes = (GetDataSize(*s.operator->()) + align - 1) & (~(align - 1));
      auto mdata = std::shared_ptr<sta::CUDAMemPool::PoolEntry>(
          new sta::CUDAMemPool::PoolEntry{static_cast<char*>(blob_mem_->addr) + off, nbytes});
      s.AssignMDataForNull(mdata);
      off += nbytes;
    }
  } else {
    for (auto &s: raw_storage_pool_) {
      s.AllocForNull(sta::MemType::kInfer, true);
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

void GraphExecutor::LoadParams(bool pipeline, bool force) {
  if (force) {
    for (auto &p : factory_.params_)
      param_ready_[p.first] = false;
  }

  for (auto &p : factory_.params_) {
    auto sid = factory_.attrs_.storage_id[p.first];
    if (!param_ready_[p.first]) {
      sta::STensor storage_tensor;
      if (Config::use_shared_tensor_infer) {
        storage_tensor = sta::TensorPool::Get()->Tensor(storage_pool_[sid]);
      } else {
        storage_tensor = raw_storage_pool_[sid];
      }
      tvm::TVMArray::CopyFromTo(
        p.second.operator->(), storage_tensor.MutableDLTensor(), load_param_stream_);
      if (pipeline) {
        ::tvm::runtime::DeviceAPI::Get(factory_.devices_[0])
            ->StreamSync(factory_.devices_[0], load_param_stream_);
        param_ready_[p.first] = true;
      }
    }
  }
  if (!pipeline) {
    ::tvm::runtime::DeviceAPI::Get(factory_.devices_[0])
        ->StreamSync(factory_.devices_[0], load_param_stream_);
    for (auto &p : factory_.params_)
      param_ready_[p.first] = true;
  }
}

void GraphExecutor::ReSetupDataEntry() {
  for (size_t i = 0; i < data_entry_.size(); i++) {
    int sid = factory_.attrs_.storage_id[i];
    // TVMArray *storage;
    // if (factory_.pool_entry_[sid].params_entry) {
    //   storage = &factory_.storage_pool_[factory_.param_node_storage_id_map_[sid]];
    // } else {
    //   storage = &storage_pool_[op_node_storage_id_map_[sid]];
    // }

    // data_entry_[i].get_mutable()->dl_tensor.data =
    //     storage_pool_[sid].get_mutable()->dl_tensor.data;
    // CHECK_NE(data_entry_[i]->data, nullptr);
    if (Config::use_shared_tensor_infer) {
      auto storage_tensor = sta::TensorPool::Get()->Tensor(storage_pool_[sid]);
      auto data_entry_view_tensor = sta::TensorPool::Get()->Tensor(data_entry_[i]);
      data_entry_view_tensor.AssignMDataForNull(storage_tensor.MData());
    } else {
      auto storage_tensor = raw_storage_pool_[sid];
      raw_data_entry_[i].AssignMDataForNull(storage_tensor.MData());
    }
  }
}

void GraphExecutor::SetupStorage(bool alloc) {
  std::vector<DLDataType> vtype;
  for (const std::string &s_type : factory_.attrs_.dltype) {
    vtype.push_back(::tvm::runtime::String2DLDataType(s_type));
  }

  size_t param_storage_size = 0;
  size_t buffer_storage_size = 0;
  for (size_t sid = 0; sid < factory_.pool_entry_.size(); sid++) {
    const auto &pit = factory_.pool_entry_[sid];
    // if (pit.params_entry)
    //   continue;
    const auto &cit = std::find_if(factory_.devices_.begin(), factory_.devices_.end(), [&pit](const DLDevice &d) {
      return pit.device_type == static_cast<int>(d.device_type);
    });
    DLDevice dev = cit == factory_.devices_.end() ? factory_.devices_[0] : *cit;
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
    if (Config::use_shared_tensor_infer) {
      if (!alloc) {
        // storage_pool_.push_back(TVMArray::Null(shape, pit.dtype, dev, mem_scope));
        storage_pool_.push_back(sta::Null(shape, pit.dtype));
      } else {
        // storage_pool_.push_back(TVMArray::Empty(shape, pit.dtype, dev, mem_scope));
        storage_pool_.push_back(sta::Empty(shape, at::MemoryFormat::Contiguous, pit.dtype, sta::MemType::kInfer));
      }
      last_tensor = sta::TensorPool::Get()->CTensor(storage_pool_.back());
    } else {
      if (!alloc) {
        raw_storage_pool_.push_back(sta::RawNull(shape, pit.dtype));
      } else {
        raw_storage_pool_.push_back(sta::RawEmpty(shape, pit.dtype, sta::MemType::kInfer));
      }
      last_tensor = raw_storage_pool_.back();
    }
    // op_node_storage_id_map_[sid] = storage_pool_.size() - 1;
    if (pit.params_entry) {
      param_storage_size += ::tvm::runtime::GetDataSize(*last_tensor.operator->());
    } else {
      buffer_storage_size += ::tvm::runtime::GetDataSize(*last_tensor.operator->());
    }
  }

  data_entry_.resize(factory_.node_row_ptr_.back());
  raw_data_entry_.resize(factory_.node_row_ptr_.back());
  data_alignment_.resize(factory_.node_row_ptr_.back());
  for (size_t i = 0; i < data_entry_.size(); i++) {
    int storage_id = factory_.attrs_.storage_id[i];
    // TVMArray* storage;
    // if (factory_.pool_entry_[storage_id].params_entry) {
    //   storage = &factory_.storage_pool_[factory_.param_node_storage_id_map_[storage_id]];
    // } else {
    //   storage = &storage_pool_[op_node_storage_id_map_[storage_id]];
    // }
    // data_entry_[i] = storage_pool_[storage_id].CreateView(factory_.attrs_.shape[i], vtype[i]);
    // const DLTensor* tmp = data_entry_[i].operator->();
    sta::STensor data_entry_view_ts;
    if (Config::use_shared_tensor_infer) {
      data_entry_[i] = sta::ViewShapeDtype(storage_pool_[storage_id], factory_.attrs_.shape[i], vtype[i]);
      data_entry_view_ts = sta::TensorPool::Get()->CTensor(data_entry_[i]);
    } else {
      raw_data_entry_[i] = sta::RawViewShapeDtype(raw_storage_pool_[storage_id], factory_.attrs_.shape[i], vtype[i]);
      data_entry_view_ts = raw_data_entry_[i];
    }
    data_alignment_[i] = details::GetDataAlignment(*data_entry_view_ts.operator->());
  }

  // setup cpu pin memory
  for (auto nid : factory_.input_nodes_) {
    if (!factory_.params_.count(nid)) {
      auto & input_id = factory_.nodes_[nid].name;
      auto & shape = factory_.attrs_.shape[nid];
      auto & dtype = factory_.attrs_.dltype[nid];
      input_cpu_pin_bufs_[input_id] = ::tvm::runtime::NDArray::Empty(
          shape, ::tvm::runtime::String2DLDataType(dtype), {kDLCUDAHost, 0});
    }
  }
  for (auto e : factory_.outputs_) {
    auto nid = entry_id(e);
    auto & output_id = factory_.nodes_[nid].name;
    auto & shape = factory_.attrs_.shape[nid];
    auto & dtype = factory_.attrs_.dltype[nid];
    output_cpu_pin_bufs_[output_id] = ::tvm::runtime::NDArray::Empty(
        shape, ::tvm::runtime::String2DLDataType(dtype), {kDLCUDAHost, 0});
  }

  static std::set<std::string> logged;
  if (!logged.count(factory_.model_name_)) {
    logged.insert(factory_.model_name_);
    LOG(INFO) << "[GraphExecutor] " << factory_.model_name_
              << " params " << 1.0 * param_storage_size / 1024 / 1024 << " Mb"
              << " intermediate " << 1.0 * buffer_storage_size / 1024 / 1024 << " Mb";
  }
}

void GraphExecutor::SetupOpExecs() {
  op_execs_.resize(factory_.nodes_.size());
  input_dltensors_.resize(factory_.node_row_ptr_.back());
  output_dltensors_.resize(factory_.node_row_ptr_.back());
  both_input_output_dltensors_.resize(factory_.node_row_ptr_.back());
  input_param_nid_.resize(op_execs_.size());
  std::unordered_set<uint32_t> input_node_eids;
  for (size_t i = 0; i < factory_.input_nodes_.size(); i++) {
    uint32_t nid = factory_.input_nodes_[i];
    input_node_eids.insert(factory_.node_row_ptr_[nid]);
  }
  std::unordered_set<uint32_t> output_node_eids;
  for (uint32_t i = 0; i < factory_.outputs_.size(); i++) {
    auto& output = factory_.outputs_[i];
    output_node_eids.insert(factory_.node_row_ptr_[output.node_id] + output.index);
  }
  
  for (uint32_t nid = 0; nid < GetNumOfNodes(); nid++) {
    const auto &inode = factory_.nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor*> args;
    for (const auto& e : inode.inputs) {
      uint32_t eid = entry_id(e);
      // args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
      sta::STensor data_entry_view_tensor;
      if (Config::use_shared_tensor_infer) {
        data_entry_view_tensor = sta::TensorPool::Get()->Tensor(data_entry_[eid]);
      } else {
        data_entry_view_tensor = raw_data_entry_[eid];
      }
      args.push_back(data_entry_view_tensor.MutableDLTensor());

      if (factory_.params_.count(e.node_id)) {
        input_param_nid_[nid].push_back(e.node_id);
      }
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; index++) {
      uint32_t eid = entry_id(nid, index);
      // args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
      sta::STensor data_entry_view_tensor;
      if (Config::use_shared_tensor_infer) {
        data_entry_view_tensor = sta::TensorPool::Get()->Tensor(data_entry_[eid]);
      } else {
        data_entry_view_tensor = raw_data_entry_[eid];
      }
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

std::pair<std::function<void()>, std::shared_ptr<OpArgs>> GraphExecutor::CreateTVMOp(
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
  ::tvm::runtime::PackedFunc pf = factory_.module_.GetFunction(param.func_name, true);
  CHECK(pf != nullptr);
  
  auto fexec = [arg_ptr, pf]() {
    ::tvm::runtime::TVMRetValue rv;
    ::tvm::runtime::TVMArgs targs(arg_ptr->arg_values.data(), arg_ptr->arg_tcodes.data(),
                                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return {fexec, arg_ptr};
}

void GraphExecutor::AllocStorageMaybeAdjust() {
  // CHECK(Config::use_shared_tensor_infer && Config::ondemand_adjust);
  CHECK(Config::ondemand_adjust);
  CHECK(!Config::infer_raw_blob_alloc);

  bool adjusted = false;

  auto adjust_train_batch_size = [this, &adjusted]() {
    if (adjusted) return;
    if (!Controller::Get()->IsTrainIdle()) {
      auto wait_train_pid = ModelTrainStore::Get()->GetTrainPid();
      this->factory_.infer_model_->SetWaitTrainPid(this->infer_model_worker_id_, wait_train_pid);
      LOG(INFO) << "[GraphExecutor] AllocStorageMaybeAdjust: model " << this->factory_.model_rank_ 
                << " wait train pid " << wait_train_pid;

      PROFILE_START(TrainAdjust, 0);
      auto cmd_id = Controller::Get()->ColocateAdjust(3);
      Controller::Get()->WaitColocateAdjustDone(cmd_id);
      PROFILE_END(TrainAdjust, 0);
    } else {
      LOG(INFO) << "[GraphExecutor] AllocStorageMaybeAdjust: model "<< this->factory_.model_rank_ << " train idle";
    }
    adjusted = true;
  };

  // ensure sequential inference allocation  
  if (!Controller::Get()->TryEnterInferModelAlloc(factory_.model_rank_)) {
    if (Controller::Get()->HasFlyingColocateAdjust()) {
      adjust_train_batch_size();
    }
    Controller::Get()->EnterInferModelAlloc(factory_.model_rank_);
  }

  double free_memory_mb;
  if (!Controller::Get()->IsTrainIdle()) {
    if (Config::use_shared_tensor_train) {
      free_memory_mb = sta::detail::ByteToMB(sta::CUDAMemPool::PoolNbytes() - sta::CUDAMemPool::InferMemUsage());
      free_memory_mb -= std::max(sta::detail::ByteToMB(sta::CUDAMemPool::TrainMemUsage()),
                                 ModelTrainStore::Get()->PredictMemUsageMB()) + Config::train_memory_over_predict_mb;
    } else {
      auto [free, total] = Profiler::GetGPUMemInfo();
      auto infer_memory_mb = sta::detail::ByteToMB(Profiler::GetLastInferMem());
      auto train_memory_mb = std::max(sta::detail::ByteToMB(Profiler::GetLastTrainMem()),
                                            ModelTrainStore::Get()->PredictMemUsageMB());
      free_memory_mb = std::min(sta::detail::ByteToMB(free), 
                                sta::detail::ByteToMB(total) - infer_memory_mb - train_memory_mb);
      free_memory_mb -= Config::train_memory_over_predict_mb; 
      DLOG(INFO) << "free " << sta::detail::ByteToMB(free) << " total " << sta::detail::ByteToMB(total)
                << " infer memory " << infer_memory_mb << " train memory " << train_memory_mb
                << " free memory " << free_memory_mb;
    }
  }

  size_t total_storage_nbytes = 0;
  std::vector<size_t> storage_nbytes(storage_pool_.size());
  for (size_t sid = 0; sid < storage_pool_.size(); sid++) {
    auto &s = storage_pool_[sid];
    auto tensor = sta::TensorPool::Get()->Tensor(s);
    auto nbytes = sta::ComputeStorageNbytes(tensor.Shape(), tensor.Stride(), 
        tensor->dtype, tensor.StorageOffset());
    storage_nbytes[sid] = nbytes;
    total_storage_nbytes += sta::detail::GetAlignedNbytes(nbytes);
  }

  if (sta::detail::ByteToMB(total_storage_nbytes) > free_memory_mb) {
    adjust_train_batch_size();
  }

  PROFILE_START(InferAllocStorage, 0);
  if (Config::use_shared_tensor_infer) {
    for (size_t sid = 0; sid < storage_pool_.size(); sid++) {
      auto &s = storage_pool_[sid];
      auto tensor = sta::TensorPool::Get()->Tensor(s);
      tensor.AllocForNull(sta::MemType::kInfer, Config::use_shared_tensor_infer ? false : true);
    }
  } else {
    for (auto &s : raw_storage_pool_) {
      s.AllocForNull(sta::MemType::kInfer, true);
    }
  }
  PROFILE_END(InferAllocStorage, 0);

  // TODO: consider fwd/bwd
  // if (sta::detail::ByteToMB(total_storage_nbytes) < free_memory_mb) {
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
  //       tensor.AssignMDataForNull(mdata);
  //       sid++;
  //     }
  //   }
  // }
  Controller::Get()->ExitInferModelAlloc(factory_.model_rank_);
}

}
}