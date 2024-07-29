#include <common/log_as_glog_sta.h>
#include <common/util.h>
#include <common/device_manager.h>
#include <common/cuda_allocator.h>
#include <common/sm_partition.h>
#include <common/xsched_ctrl.h>
#include <common/inf_tra_comm/communicator.h>

#include <torch_col/csrc/init.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/torch_allocator_plugin.h>
#include <torch_col/csrc/dist_ext.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>

#include <Python.h>
#include <pybind11/pybind11.h>


namespace torch_col {
  
void TorchColInit(int train_rank, int train_world_size) {
  // first init config before call any other functions
  TorchColConfig::InitConfig(train_rank, train_world_size);

  COL_NVML_CALL(nvmlInit());
  colserve::sta::DeviceManager::Init();
  torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
  if (TorchColConfig::IsEnableSharedTensor()) {
    // we assume one training process one gpu 
    torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()->init(train_world_size);
    torch::cuda::CUDAColAllocator::CUDAColAllocator::SetCurrentAllocator();
  }


  if (TorchColConfig::HasColocatedInferServer()) {
    colserve::ctrl::InfTraCommunicator::Init(false, false, 
                                             train_world_size);
  }
  TorchExtInit();

  LOG(INFO) << "TorchCol initialized.";

  CUDA_CALL(cudaSetDevice(TorchColConfig::GetTrainRank()));
  CUDA_CALL(cudaDeviceSynchronize());
}

void SMPartitionInit(uint64_t stream) {
  if (!TorchColConfig::dynamic_sm_partition) {
    return ;
  }  

  colserve::SMPartitioner::Init(TorchColConfig::GetTrainRank());

  // auto stream = reinterpret_cast<cudaStream_t>(
  //   colserve::sta::xsched::GetRegisteredGlobalStream());
  // CHECK(reinterpret_cast<uint64_t>(stream) != 0);

  LOG(INFO) << "Init SMPartition, stream " << stream << " " << *(void**)stream;

  auto hook = [stream]() -> void*{
    // colserve::SetGlobalTPCMask(0x1);
    // colserve::SetStreamTpcMask(stream, 0x1);
    auto mask = colserve::SMPartitioner
        ::Get(TorchColConfig::GetTrainRank())
        ->SetTrainStreamTpcMask(reinterpret_cast<cudaStream_t>(stream));
    // LOG(INFO) << "set train stream tpc mask " << std::hex << mask;
    return nullptr;
  };

  auto succ = colserve::sta::xsched::RegisterCudaKernelLaunchPreHook(hook);
  if (!succ) {
    LOG(FATAL) << "[PySched] RegisterCudaKernelLaunchPreHook failed"; 
  }
  LOG(INFO) << "[TorchColInit] init_sm_partition done";
}


namespace {
template<typename T>
using shared_ptr_class_ = pybind11::class_<T, std::shared_ptr<T>>;
}


void TorchExtInit() {
  auto c_module = pybind11::module::import("torch_col._C");
  auto module = c_module.def_submodule("_dist");
  // auto torch_c_dist_model = pybind11::module::import("torch._C._distributed_c10d");

  namespace py = pybind11;
  namespace jit = torch::jit;

  /////////////////////////////////////////////
  // define ext class

  // Reducer
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
        "_set_grads_to_none",
        [](::c10d::Reducer& reducer) { reducer.set_grads_to_none(true); },
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
        "set_logger",
        [](Reducer& reducer,
            const std::shared_ptr<Logger> logger) {
          std::weak_ptr<::c10d::Logger> logger_weakref = logger;
          reducer.set_logger(logger_weakref);
        })
    /* extend the Reducer */
    .def(
        "finalize_dropped_batch",
        &Reducer::finalize_dropped_batch,
        py::call_guard<py::gil_scoped_release>());


  // Logger
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


  /////////////////////////////////////////////
  // override torch distributed Reducer

  auto torch_dist_module = pybind11::module::import("torch.distributed");
  torch_dist_module.attr("Reducer") = module.attr("Reducer");
  torch_dist_module.attr("Logger") = module.attr("Logger");

  pybind11::module::import("torch._C._distributed_c10d")
    .attr("Reducer") = module.attr("Reducer");
}

} // namespace torch_col
