#include <common/log_as_glog_sta.h>
#include <common/util.h>

#include <torch_col/csrc/dist_ext.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/utils/pybind.h>

#include <boost/format.hpp>

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>


namespace torch_col {

::c10::intrusive_ptr<ProcessGroupNCCL> ProcessGroupNCCL::default_pg_ = nullptr;

void Reducer::finalize_dropped_batch() {
  // ::c10d::Reducer::finalize_backward();
  LOG(INFO) << "Reducer::finalize_dropped_batch() begin";
  if (expect_autograd_hooks_) {
    // ::c10d::Reducer::finalize_backward();
    expect_autograd_hooks_ = false;
    require_finalize_ = false;
    for (auto & bucket : buckets_) {
      bucket.future_work->wait();
      div_factor_ = -1; // kUnsetDivFactor = -1;
    }
  } else {
    require_finalize_ = false;
  }
  LOG(INFO) << "Reducer::finalize_dropped_batch() end";
}

std::vector<ncclComm_t> ProcessGroupNCCL::GetNcclComm(
    const std::vector<at::Device> &devices) const {
  auto device_key = getKeyFromDevices(devices);
  auto it = devNCCLCommMap_.find(device_key);
  if (it == devNCCLCommMap_.end()) {
    return {};
  }

  std::vector<ncclComm_t> comms;
  for (auto & comm : it->second) {
    comms.push_back(comm->getNcclComm());
  }
  return comms;
}

::c10::intrusive_ptr<ProcessGroupNCCL> 
ProcessGroupNCCL::GetDefaultProcessGroupNCCL() {
  return default_pg_;
}

void ProcessGroupNCCL::SetDefaultProcessGroupNCCL(
      ProcessGroupNCCL *pg) {
  default_pg_ = 
      ::c10::intrusive_ptr<ProcessGroupNCCL>::unsafe_steal_from_new(pg);
}

void ProcessGroupNCCL::SetDefaultProcessGroupNCCL(
    const ::c10::intrusive_ptr<ProcessGroupNCCL> &pg) {
  default_pg_ = pg;
}

ProcessGroupNCCL::ProcessGroupNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
  : ::c10d::ProcessGroupNCCL(store, rank, size, options) {}

void ProcessGroupNCCL::RestartNcclComm(
    const std::vector<at::Device> &devices) {
  auto device_key = getKeyFromDevices(devices);
  std::vector<std::shared_ptr<::c10d::NCCLComm>> nccl_comms;

  std::unique_lock<std::mutex> lock(mutex_);
  {
    auto it = devNCCLCommMap_.find(device_key);
    if (it == devNCCLCommMap_.end()) {
      std::stringstream ss;
      for(auto & device : devices) {
        ss << device << " ";
      }
      LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
                << " device " << ss.str()
                << " device_key " << device_key
                << " not found, valid keys: "
                << GetDevNcclCommMapKeySetStrsUnlocked();
      return;
    }
    nccl_comms = it->second;
  }

  LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
            << " find device_key " << device_key
            << " nccl_comms.size() " << nccl_comms.size();

  std::vector<std::pair<uint32_t, uint32_t>> abort_flags(nccl_comms.size());
  for (int i = 0; i < nccl_comms.size(); i++) {
    auto & comm = nccl_comms[i];
    abort_flags[i] = _GetNcclCommAbortFlag(comm->getNcclComm());
  }


  bool is_abort_flag_setted = (abort_flags[0].first || abort_flags[0].second);
  for (int i = 1; i < abort_flags.size(); i++) {
    auto & flag = abort_flags[i];
    if ((flag.first || flag.second) != is_abort_flag_setted) {
      LOG(FATAL) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
                << " device_key " << device_key
                << " abort flags are not the same";
      break;
    }
  }

  if (is_abort_flag_setted) {
    RestartNcclCommByReSettingAbortFlag(devices, device_key, nccl_comms, lock);
  } else {
    RestartNcclCommByRecreating(devices, device_key, nccl_comms, lock);
  }
  return ;

  ///////

  auto old_ncclId = nccl_comms[0]->getNcclId();
  
  // lock the nccl work to avoid watch dog detecting the error
  // during the restart nccl comm
  {
    std::unique_lock<std::mutex> work_meta_list_lock(workMetaListMutex_);

    // destroyNCCLComms()
    // for (auto & comm : nccl_comms) {
    //   comm->ncclCommAbort();
    // }
    for (auto & comm : nccl_comms) {
      _SetNcclCommAbortFlag(comm->getNcclComm(), 0);
      // comm->ncclCommAbort();
    }
    
    // remove work in list, rely on cuda stream to ensure the work is done
    workMetaList_.clear();
  }

  // devNCCLCommMap_.erase(device_key);

  // sync the nccl stream to ensure all the nccl work is completed
  auto nccl_streams_it = ncclStreams_.find(device_key);
  CHECK(nccl_streams_it != ncclStreams_.end());
  for (auto & stream : nccl_streams_it->second) {
    COL_CUDA_CALL(cudaStreamSynchronize(stream));
  }

  return;
  // RestartNcclCommByRecreating(devices, device_key, nccl_comms, lock);


  // // re-create NCCL comms, we DO NOT want nccl streams to be changed,
  // // thus, we manually create the new nccl comms
  // ncclUniqueId ncclId{0};
  // if (getRank() == 0) {
  //   ncclGetUniqueId(&ncclId);
  // }
  // broadcastUniqueNCCLID(&ncclId, false, device_key, 0);
  // LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
  //           << " new ncclId " << buildNcclUniqueIdStr(ncclId);

  // COL_NCCL_CALL(ncclGroupStart());
  // for (auto & comm : nccl_comms) {
  //   comm = ::c10d::NCCLComm::create(size_, rank_, ncclId);
  // }
  // COL_NCCL_CALL(ncclGroupEnd());

  // ncclIdToCommMap_.emplace(buildNcclUniqueIdStr(ncclId), nccl_comms);
  // devNCCLCommMap_[device_key] = nccl_comms;

  // LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
  //           << " device_key " << device_key
  //           << " re-create NCCL comms done";

  // // re-create NCCL comms
  // getNCCLComm(device_key, devices, c10d::OpType::ALLREDUCE);

  // for (auto & comm : nccl_comms) {
  //   ncclComm_t nccl_comm = comm->getNcclComm();
  //   ncclCommAbort(nccl_comm);
  //   // ncclCommInitRank(&nccl_comm, )
  // }

  // ncclUniqueId nccl_id;
  // if (getRank() == 0) {
  //   ncclGetUniqueId(&nccl_id);
  // }
  // broadcastUniqueNCCLID(&nccl_id, false, device_key, 0);

  // ncclGroupStart();
  // for (int i = 0; i < nccl_comms.size(); i++) {
  //   auto & comm = nccl_comms[i];
  //   ncclComm_t nccl_comm;

  //   int nRank = devices.size() * getSize();
  //   int rank = getRank() * devices.size() + i;
  //   comm = ::c10d::NCCLComm::create(nRank, rank, nccl_id);
  // }
  // ncclGroupEnd();

  
  
  // // 1. abort and restart the nccl comm
  // for (auto & comm : nccl_comms) {
  //   ncclComm_t nccl_comm = comm->getNcclComm();
  //   ncclCommAbort(nccl_comm);
  //   ncclCommInitRank(&nccl_comm, )
  // }
}

void ProcessGroupNCCL::RestartNcclCommByReSettingAbortFlag(      
    const std::vector<at::Device> &devices, 
    const std::string &device_key,
    std::vector<std::shared_ptr<::c10d::NCCLComm>> &comms,
    const std::unique_lock<std::mutex> &pg_lock) {
  {
    std::unique_lock<std::mutex> work_meta_list_lock(workMetaListMutex_);
    workMetaList_.clear();
  }

  auto nccl_streams_it = ncclStreams_.find(device_key);
  CHECK(nccl_streams_it != ncclStreams_.end());
  for (auto & stream : nccl_streams_it->second) {
    COL_CUDA_CALL(cudaStreamSynchronize(stream));
  }

  for (auto & comm : comms) {
    _SetNcclCommAbortFlag(comm->getNcclComm(), 0);
  }

  LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
            << " device_key " << device_key
            << " re-set abort flag done";
}

void ProcessGroupNCCL::RestartNcclCommByRecreating(
    const std::vector<at::Device> &devices, 
    const std::string &device_key,
    std::vector<std::shared_ptr<::c10d::NCCLComm>> &nccl_comms,
    const std::unique_lock<std::mutex> &pg_lock) {
  
  auto old_ncclId = nccl_comms[0]->getNcclId();
  
  // lock the nccl work to avoid watch dog detecting the error
  // during the restart nccl comm
  {
    std::unique_lock<std::mutex> work_meta_list_lock(workMetaListMutex_);

    for (auto & comm : nccl_comms) {
      comm->ncclCommAbort();
    }
    
    // remove work in list, rely on cuda stream to ensure the work is done
    workMetaList_.clear();
  }

  // devNCCLCommMap_.erase(device_key);

  // sync the nccl stream to ensure all the nccl work is completed
  auto nccl_streams_it = ncclStreams_.find(device_key);
  CHECK(nccl_streams_it != ncclStreams_.end());
  for (auto & stream : nccl_streams_it->second) {
    COL_CUDA_CALL(cudaStreamSynchronize(stream));
  }

  // re-create NCCL comms, we DO NOT want nccl streams to be changed,
  // thus, we manually create the new nccl comms
  ncclUniqueId ncclId{0};
  if (getRank() == 0) {
    ncclGetUniqueId(&ncclId);
  }
  broadcastUniqueNCCLID(&ncclId, false, device_key, 0);
  LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
            << " new ncclId " << buildNcclUniqueIdStr(ncclId);

  COL_NCCL_CALL(ncclGroupStart());
  for (auto & comm : nccl_comms) {
    comm = ::c10d::NCCLComm::create(size_, rank_, ncclId);
  }
  COL_NCCL_CALL(ncclGroupEnd());

  ncclIdToCommMap_.emplace(buildNcclUniqueIdStr(ncclId), nccl_comms);
  devNCCLCommMap_[device_key] = nccl_comms;

  LOG(INFO) << str(boost::format("[Rank %d | RestartNcclComm]") % getRank())
            << " device_key " << device_key
            << " re-create NCCL comms done";
}

void ProcessGroupNCCL::SetNcclCommAbortFlag(
    const std::vector<at::Device> &devices,
    uint32_t val) {
  auto device_key = getKeyFromDevices(devices);
  std::vector<std::shared_ptr<::c10d::NCCLComm>> ncc_comms;

  std::unique_lock<std::mutex> lock(mutex_);
  auto it = devNCCLCommMap_.find(device_key);
  if (it == devNCCLCommMap_.end()) {
    LOG(INFO) << str(boost::format("[Rank %d | SetNcclCommAbortFlag]") % getRank())
              << " device_key " << device_key
              << " not found, valid keys: "
              << GetDevNcclCommMapKeySetStrsUnlocked();
    return;
  }
  ncc_comms = it->second;

  for (auto & comm : ncc_comms) {
    _SetNcclCommAbortFlag(comm->getNcclComm(), val);
  }
  LOG(INFO) << str(boost::format("[Rank %d | SetNcclCommAbortFlag]") % getRank())
            << " device_key " << device_key
            << " set abort flag done";
}

void ProcessGroupNCCL::_SetNcclCommAbortFlag(ncclComm_t comm, uint32_t val) {
  auto [abort_flag, child_abort_flag] = _GetNcclCommAbortFlagPtr(comm);
  *abort_flag = val;
  *child_abort_flag = val;
  this->abort_flag_ = val;
}

std::pair<uint32_t, uint32_t> 
ProcessGroupNCCL::_GetNcclCommAbortFlag(ncclComm_t comm) {
  auto [abort_flag, child_abort_flag] = _GetNcclCommAbortFlagPtr(comm);
  return {*abort_flag, *child_abort_flag};
}

std::pair<uint32_t*, uint32_t*> 
ProcessGroupNCCL::_GetNcclCommAbortFlagPtr(ncclComm_t comm) {
  auto nccl_version = c10d::getNcclVersion();

  if (nccl_version == "2.18.6") {
    auto abort_flag_offset = 10784;
    uint32_t* abort_flag = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(comm) + abort_flag_offset);

    auto child_abort_flag_offset = 10792;
    uint32_t* child_abort_flag = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(comm) + child_abort_flag_offset);

    return std::make_pair(abort_flag, child_abort_flag);
  } else {
    LOG(FATAL) << "NCCL version " << nccl_version << " not supported"
               << " abort flag offset may be different";
    return {};
  }
}

std::string ProcessGroupNCCL::GetDevNcclCommMapKeySetStrs() {
  std::unique_lock lock(mutex_);
  return GetDevNcclCommMapKeySetStrsUnlocked();
}

std::string ProcessGroupNCCL::GetDevNcclCommMapKeySetStrsUnlocked() const {
  std::string key_set_strs =
      std::accumulate(devNCCLCommMap_.begin(), devNCCLCommMap_.end(),
        std::string{},
        [](std::string &acc, auto &kv) {
          return acc.empty() ? kv.first : acc + ", " + kv.first;
        });
  return "{ " + key_set_strs + " }";
}

} // namespace torch_col



///////////////////////////////////////////////////////////////////////////////
// MARK: Overwrite the following functions of c10d

namespace {
// from torch/csrc/distributed/c10d/init.cpp

// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_;

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

}

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);

namespace torch_col {

namespace py = ::pybind11;
namespace jit = torch::jit;

namespace {

template<typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

}

void TorchExtInit() {
  auto c_module = py::module::import("torch_col._C");
  auto module = c_module.def_submodule("_dist");
  // auto torch_c_dist_model = pybind11::module::import("torch._C._distributed_c10d");

  /////////////////////////////////////////////////////////////////////////////
  // define ext class

  
  try {
    // Reducer
    LOG(INFO) << "torch_col::TorchExtInit() Reducer";
    shared_ptr_class_<Reducer>(module, "Reducer")
    .def(
        py::init<
            std::vector<at::Tensor>,
            std::vector<std::vector<size_t>>,
            std::vector<size_t>,
            c10::intrusive_ptr<::c10d::ProcessGroup>,
            std::vector<bool>,
            int64_t,
            bool,
            bool,
            std::unordered_map<size_t, std::string>,
            int64_t>(),
        py::arg("params"),
        py::arg("bucket_indices"),
        py::arg("per_bucket_size_limits"),
        py::arg("process_group"),
        py::arg("expect_sparse_gradients") = std::vector<bool>(),
        py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
        py::arg("find_unused_parameters") = false,
        py::arg("gradient_as_bucket_view") = false,
        py::arg("param_to_name_mapping") =
            std::unordered_map<size_t, std::string>(),
        py::arg("first_bucket_bytes_cap") = ::c10d::kDefaultFirstBucketBytes,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "prepare_for_forward",
        &::c10d::Reducer::prepare_for_forward,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "prepare_for_backward",
        &::c10d::Reducer::prepare_for_backward,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "prepare_for_backward",
        [](::c10d::Reducer& reducer, const at::Tensor& output) -> void {
        reducer.prepare_for_backward({output});
        },
        py::call_guard<py::gil_scoped_release>())
    .def("get_backward_stats", &::c10d::Reducer::get_backward_stats)
    .def(
        "_install_post_backward_futures",
        [](::c10d::Reducer& reducer,
            const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>&
                futs) {
        c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
            c10::FutureType::create(c10::TensorType::get()));
        for (const auto& fut : futs) {
            futures.push_back(fut->fut);
        }
        reducer.install_futures(std::move(futures));
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_rebuild_buckets",
        &::c10d::Reducer::rebuild_buckets,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_get_zeros_like_grad_buckets",
        [](::c10d::Reducer& reducer) {
        return reducer.get_grad_buckets(/* return_zero_tensors */ true);
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_optimizer_in_backward",
        [](::c10d::Reducer& reducer) { reducer.set_optimizer_in_backward(); },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_sparse_metadata",
        &::c10d::Reducer::setSparseMetadata,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_mixed_precision_param_dtype",
        [](::c10d::Reducer& reducer, py::object data_type_obj) {
        auto scalar_type =
            reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
        reducer.set_mixed_precision_param_dtype(scalar_type);
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_push_all_rebuilt_params",
        &::c10d::Reducer::push_rebuilt_params_for_all_indices,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_forward_pass_work_handle",
        &::c10d::Reducer::set_forward_pass_work_handle,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_get_local_used_map", &::c10d::Reducer::get_local_used_map_on_device)
    .def(
        "_set_ddp_runtime_logging_sample_rate",
        &::c10d::Reducer::set_ddp_runtime_logging_sample_rate,
        py::arg("sample_rate"),
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_static_graph",
        &::c10d::Reducer::set_static_graph,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_ddp_graph_static",
        &::c10d::Reducer::ddp_graph_static,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_delay_all_reduce",
        &::c10d::Reducer::delay_all_reduce,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_run_comm_hook",
        [](::c10d::Reducer& reducer, ::c10d::GradBucket& bucket)
            -> std::shared_ptr<jit::PythonFutureWrapper> {
        c10::intrusive_ptr<c10::ivalue::Future> fut =
            reducer.run_comm_hook(bucket);
        return std::make_shared<jit::PythonFutureWrapper>(fut);
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_run_allreduce_hook",
        [](::c10d::Reducer& reducer, ::c10d::GradBucket& bucket)
            -> std::shared_ptr<jit::PythonFutureWrapper> {
        c10::intrusive_ptr<c10::ivalue::Future> fut =
            reducer.run_allreduce_hook(bucket);
        return std::make_shared<jit::PythonFutureWrapper>(fut);
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_autograd_hook",
        [](::c10d::Reducer& reducer, int index) -> void {
        reducer.autograd_hook(index);
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "set_logger",
        [](Reducer& reducer,
            const std::shared_ptr<Logger> logger) {
        std::weak_ptr<::c10d::Logger> logger_weakref = logger;
        reducer.set_logger(logger_weakref);
        })
    .def(
        "_remove_autograd_hooks",
        [](::c10d::Reducer& reducer) { reducer.remove_autograd_hooks(); },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_check_reducer_finalized",
        [](::c10d::Reducer& reducer) { return reducer.check_finalized(); },
        py::call_guard<py::gil_scoped_release>())
    /* extend the Reducer */
    .def(
        "finalize_dropped_batch",
        &Reducer::finalize_dropped_batch,
        py::call_guard<py::gil_scoped_release>());


    // Logger
    LOG(INFO) << "torch_col::TorchExtInit() Logger";
    shared_ptr_class_<Logger>(module, "Logger")
    .def(
        py::init<std::shared_ptr<Reducer>>(),
        py::arg("reducer"),
        py::call_guard<py::gil_scoped_release>())
    .def(
        "set_construction_data_and_log",
        &::c10d::Logger::set_construction_data_and_log,
        py::arg("module_name"),
        py::arg("device_ids"),
        py::arg("output_device"),
        py::arg("broadcast_buffers"),
        py::arg("has_sync_bn"),
        py::arg("static_graph"),
        py::call_guard<py::gil_scoped_release>())
    .def(
        "set_runtime_stats_and_log",
        &::c10d::Logger::set_runtime_stats_and_log,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "set_error_and_log",
        [](::c10d::Logger& logger, const std::string& error) {
        logger.set_error_and_log(error);
        },
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_get_ddp_logging_data",
        &::c10d::Logger::get_ddp_logging_data,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_comm_hook_name",
        &::c10d::Logger::set_comm_hook,
        py::arg("comm_hook"),
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_uneven_input_join",
        &::c10d::Logger::set_uneven_input_join,
        py::call_guard<py::gil_scoped_release>())
    .def(
        "_set_static_graph",
        &::c10d::Logger::set_static_graph,
        py::call_guard<py::gil_scoped_release>());

    
    // ProcessGroupNCCL
    LOG(INFO) << "torch_col::TorchExtInit() ProcessGroupNCCL";
    auto backend = py::module::import("torch._C._distributed_c10d").attr("Backend");
    LOG(INFO) << "torch_col::TorchExtInit() ProcessGroupNCCL backend " << backend.ptr();
    auto processGroupNCCL =
      intrusive_ptr_no_gil_destructor_class_<ProcessGroupNCCL>(
      module, "ProcessGroupNCCL", backend)
      .def(
          py::init<
              const c10::intrusive_ptr<::c10d::Store>&,
              int,
              int,
              c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options>>(),
          py::call_guard<py::gil_scoped_release>())
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      const std::chrono::milliseconds& timeout) {
            auto options = ::c10d::ProcessGroupNCCL::Options::create();
            options->is_high_priority_stream = false;
            options->timeout = timeout;
            return c10::make_intrusive<ProcessGroupNCCL>(
                store, rank, size, options);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = kProcessGroupDefaultTimeout,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_abort",
          [](const c10::intrusive_ptr<ProcessGroupNCCL>& self,
              const c10::optional<std::string>& abortReason) {
            return self->abort(abortReason);
          },
          py::arg("abort_reason") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def("_group_start", &::c10d::ProcessGroupNCCL::groupStart)
      .def("_group_end", &::c10d::ProcessGroupNCCL::groupEnd)
      .def_property_readonly(
          "options", &::c10d::ProcessGroupNCCL::getOptions)
      .def_property_readonly(
          "is_ucc_available", &::c10d::ProcessGroupNCCL::isUCCAvailable)
      /* ext ProcessGroupNCCL */
      .def("_restart_nccl_comm", &ProcessGroupNCCL::RestartNcclComm,
          py::arg("devices"),
          py::call_guard<py::gil_scoped_release>())
      .def("_set_nccl_comm_abort_flag", &ProcessGroupNCCL::SetNcclCommAbortFlag,
          py::arg("devices"),
          py::arg("val") = 1,
          py::call_guard<py::gil_scoped_release>())
      .def("_set_as_default_pg", 
          [](const c10::intrusive_ptr<ProcessGroupNCCL> &self) {
            ProcessGroupNCCL::SetDefaultProcessGroupNCCL(self);
          },
          py::call_guard<py::gil_scoped_release>());

    // ProcessGroupNCCL.Options
    auto c10d_processGroupNcclOptions = 
        py::module::import("torch._C._distributed_c10d")
        .attr("ProcessGroupNCCL")
        .attr("Options");
    processGroupNCCL.attr("Options") = c10d_processGroupNcclOptions;

    LOG(INFO) << "torch_col::TorchExtInit() register python objects done";
  } catch (const std::exception& e) {
    LOG(FATAL) << "torch_col::TorchExtInit() exception " << e.what();
  }

  /////////////////////////////////////////////////////////////////////////////
  // overwrite py-torch distributed module attributes

  auto torch_c_dist_module = py::module::import("torch._C._distributed_c10d");
  torch_c_dist_module.attr("Reducer") = module.attr("Reducer");
  torch_c_dist_module.attr("Logger") = module.attr("Logger");
  torch_c_dist_module.attr("ProcessGroupNCCL") = module.attr("ProcessGroupNCCL");

  auto torch_dist_module = py::module::import("torch.distributed");
  torch_dist_module.attr("Reducer") = module.attr("Reducer");
  torch_dist_module.attr("Logger") = module.attr("Logger");
  torch_dist_module.attr("ProcessGroupNCCL") = module.attr("ProcessGroupNCCL");

  auto torch_c10d_module = torch_dist_module.attr("distributed_c10d");
  torch_c10d_module.attr("ProcessGroupNCCL") = module.attr("ProcessGroupNCCL");
  
  LOG(INFO) << "torch_col::TorchExtInit() DONE";
}

} // namespace torch_col


///////////////////////////////////////////////////////////////////////////////
// torch functions in ::c10d, as they are LOCAL functions,
// we re-implement them

// from torch/csrc/distributed/c10d/NCCLUtils.cpp
namespace c10d {

ncclComm_t NCCLComm::getNcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    auto commFailureMsg = commFailureReason_ != c10::nullopt
        ? c10::str(" Original reason for failure was: ", *commFailureReason_)
        : "";
    TORCH_CHECK(
        false,
        c10::str(
            "NCCL communicator was aborted on rank ",
            rank_,
            ". ",
            commFailureMsg));
  }
  return ncclComm_;
}

std::string getNcclVersion() {
  static c10::once_flag ncclGetVersionFlag;
  static std::string versionString;

  c10::call_once(ncclGetVersionFlag, []() {
    int version;
    ncclResult_t status = ncclGetVersion(&version);
    // can't compute the version if call did not return successfully or version
    // code < 100 (corresponding to 0.1.0)
    if (status != ncclSuccess || version < 100) {
      versionString = "Unknown NCCL version";
    } else {
      // NCCL changed version coding starting 2.9
      const int majorBase = version < 2900 ? 1000 : 10000;
      const int minorBase = 100;
      auto ncclMajor = version / majorBase;
      auto ncclMinor = (version % majorBase) / minorBase;
      auto ncclPatch =
          version % (ncclMajor * majorBase + ncclMinor * minorBase);
      versionString = std::to_string(ncclMajor) + "." +
          std::to_string(ncclMinor) + "." + std::to_string(ncclPatch);
    }
  });

  return versionString;
}

std::string ncclGetErrorWithVersion(ncclResult_t error) {
  return std::string(ncclGetErrorString(error)) + ", NCCL version " +
      getNcclVersion();
}

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
std::string getNcclErrorDetailStr(
    ncclResult_t error,
    c10::optional<std::string> processGroupFailureReason /* = c10::nullopt */
) {
  // Prioritize failure reason provided by PG NCCL first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != c10::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
#ifdef ENABLE_NCCL_GET_LAST_ERROR
  err = "\nLast error:\n" + std::string(ncclGetLastError(NULL));
#endif
  switch (error) {
    case ncclUnhandledCudaError:
      interpret = "ncclUnhandledCudaError: Call to CUDA function failed.";
      break;
    case ncclSystemError:
      interpret =
          "ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. ";
#ifndef NCCL_REMOTE_ERROR
      // Before ncclRemoteError was created, unexpected remote disconnect was
      // categorized as ncclSystemError
      interpret += "It can be also caused by unexpected exit of a remote peer.";
#endif
      break;
    case ncclInternalError:
      interpret = "ncclInternalError: Internal check failed.";
      break;
    case ncclInvalidArgument:
      interpret = "ncclInvalidArgument: Invalid value for an argument.";
      break;
    case ncclInvalidUsage:
      interpret =
          "ncclInvalidUsage: This usually reflects invalid usage of NCCL library.";
      break;
#ifdef NCCL_REMOTE_ERROR
    case ncclRemoteError:
      interpret =
          "ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.";
      break;
#endif
    default:
      interpret = "Unknown NCCL error!";
  }
  return interpret + err;
}

}