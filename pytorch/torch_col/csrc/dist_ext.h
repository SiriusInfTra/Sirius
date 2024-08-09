#pragma once

#include <torch/csrc/distributed/c10d/reducer.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

namespace torch_col {

class Reducer : public ::c10d::Reducer {
 public:
  template<typename ... T>
  Reducer(T&& ... args) : 
      ::c10d::Reducer(std::forward<T>(args)...) {}

  void finalize_dropped_batch();
 private:
};

class Logger : public ::c10d::Logger {
 public:
  template<typename ... T>
  Logger(T&& ... args) : 
      ::c10d::Logger(std::forward<T>(args)...) {}
};

class ProcessGroupNCCL : public ::c10d::ProcessGroupNCCL {
 public:
  class WorkDummy : public ::c10d::Work {
   public:
    WorkDummy(int rank = -1, 
              ::c10d::OpType opType = ::c10d::OpType::UNKNOWN)
        : ::c10d::Work(rank, opType) {
      completed_ = true;      
    }

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      auto future = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
      future->markCompleted();
      return future;
    }
  };

  static ::c10::intrusive_ptr<ProcessGroupNCCL> GetDefaultProcessGroupNCCL();
  static void SetDefaultProcessGroupNCCL(ProcessGroupNCCL *pg);
  static void SetDefaultProcessGroupNCCL(const ::c10::intrusive_ptr<ProcessGroupNCCL> &pg);

  // from torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  ProcessGroupNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options = Options::create());

  // This constructor includes the deprecated `groupName` argument.
  // If you have existing code that uses the `groupName`, you can replace
  // it by specifying a `c10d::PrefixStore(groupName, store)` for store.
  C10_DEPRECATED ProcessGroupNCCL(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::string& groupName,
      c10::intrusive_ptr<Options> options = Options::create())
      : ProcessGroupNCCL(store, rank, size, options) {}
  // end from torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp

  ~ProcessGroupNCCL() override {
    // recover the abort flag to let nccl release the resources
    CHECK(abort_flag_ == 0) << "abort flag should be 0";
  }

  /////////////////////////////////////////////////////////////////////////////
  // override the original collective functions to handle the abort flag
  #define REDISPATCH_COLLECTIVE_FUNC(op_type, func, args...) \
    do { \
      if (abort_flag_ != 0) { \
        LOG(INFO) << "[ProcessGroupNCCL | REDISPATCH_COLLECTIVE_FUNC] abort " << #func;  \
        return ::c10::make_intrusive<WorkDummy>(rank_, op_type); \
      } else { \
        return func(args); \
      } \
    } while (0)

  ::c10::intrusive_ptr<::c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const ::c10d::BroadcastOptions& opts = ::c10d::BroadcastOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(::c10d::OpType::BROADCAST, 
      ::c10d::ProcessGroupNCCL::broadcast, tensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> allreduce_sparse(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceOptions& opts = ::c10d::AllreduceOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(::c10d::OpType::ALLREDUCE, 
      ::c10d::ProcessGroupNCCL::allreduce_sparse, tensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceOptions& opts = ::c10d::AllreduceOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(::c10d::OpType::ALLREDUCE, 
      ::c10d::ProcessGroupNCCL::allreduce, tensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const ::c10d::AllreduceCoalescedOptions& opts =
          ::c10d::AllreduceCoalescedOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::ALLREDUCE_COALESCED, 
        ::c10d::ProcessGroupNCCL::allreduce_coalesced, tensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ::c10d::ReduceOptions& opts = ::c10d::ReduceOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::REDUCE, 
        ::c10d::ProcessGroupNCCL::reduce, tensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllgatherOptions& opts = ::c10d::AllgatherOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::ALLGATHER, 
        ::c10d::ProcessGroupNCCL::allgather, 
        outputTensors, inputTensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllgatherOptions& opts = ::c10d::AllgatherOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::ALLGATHER_COALESCED, 
        ::c10d::ProcessGroupNCCL::allgather_coalesced, 
        outputTensorLists, inputTensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ::c10d::AllgatherOptions& opts = ::c10d::AllgatherOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::ALLGATHER_COALESCED, 
        ::c10d::ProcessGroupNCCL::allgather_into_tensor_coalesced, 
        outputs, inputs, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ::c10d::ReduceScatterOptions& opts = ::c10d::ReduceScatterOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::REDUCE_SCATTER, 
        ::c10d::ProcessGroupNCCL::reduce_scatter, 
        outputTensors, inputTensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ::c10d::ReduceScatterOptions& opts = ::c10d::ReduceScatterOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::REDUCE_SCATTER, 
        ::c10d::ProcessGroupNCCL::reduce_scatter_tensor_coalesced, 
        outputs, inputs, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> barrier(
      const ::c10d::BarrierOptions& opts = ::c10d::BarrierOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::BARRIER, 
        ::c10d::ProcessGroupNCCL::barrier, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const ::c10d::AllToAllOptions& opts = ::c10d::AllToAllOptions()) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::ALLTOALL, 
        ::c10d::ProcessGroupNCCL::alltoall, outputTensors, inputTensors, opts);
  }

  ::c10::intrusive_ptr<::c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::SEND, 
        ::c10d::ProcessGroupNCCL::send, tensors, dstRank, tag);
  }

  ::c10::intrusive_ptr<::c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    REDISPATCH_COLLECTIVE_FUNC(
        ::c10d::OpType::RECV, 
        ::c10d::ProcessGroupNCCL::recv, tensors, srcRank, tag);
  }
  /////////////////////////////////////////////////////////////////////////////

  std::vector<ncclComm_t> GetNcclComm(
      const std::vector<at::Device> &device_key) const;

  void RestartNcclComm(
      const std::vector<at::Device> &devices);

  void SetNcclCommAbortFlag(const std::vector<at::Device> &devices, uint32_t val = 1 /* default to abort */);

 private:
  // from torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  // but as the static methods
  static std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
    std::string deviceList;
    for (auto& device : devices) {
      if (deviceList.empty()) {
        deviceList = std::to_string(device.index());
      } else {
        deviceList += "," + std::to_string(device.index());
      }
    }
    return deviceList;
  }

  // from torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  // Get the list of devices from list of tensors
  std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
    std::vector<at::Device> res;
    res.reserve(tensors.size());
    for (auto& tensor : tensors) {
      // tensors must all be on the same device, or all on distinct devices.
      // The line below assumes that constraint has already been enforced
      // (by check_gpu_tensors_same_device or
      // check_gpu_tensors_different_devices).
      if (res.size() == 0 || tensor.device() != res[0]) {
        res.push_back(tensor.device());
      }
    }
    return res;
  }

  // from torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  // Given a ncclUniqueId, convert it to a string representation that can be put
  // in the store.
  static std::string buildNcclUniqueIdStr(const ncclUniqueId& ncclID) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
    std::ostringstream oss;
    for (const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
      oss << std::hex << static_cast<int>(bytes[i]);
    }
    return oss.str();
  }

  void RestartNcclCommByReSettingAbortFlag(
      const std::vector<at::Device> &devices, 
      const std::string &device_key,
      std::vector<std::shared_ptr<::c10d::NCCLComm>> &nccl_comms,
      const std::unique_lock<std::mutex> &pg_lock);
  void RestartNcclCommByRecreating(
      const std::vector<at::Device> &devices, 
      const std::string &device_key,
      std::vector<std::shared_ptr<::c10d::NCCLComm>> &nccl_comms,
      const std::unique_lock<std::mutex> &pg_lock);

  void _SetNcclCommAbortFlag(ncclComm_t comm, uint32_t val);
  std::pair<uint32_t, uint32_t> _GetNcclCommAbortFlag(ncclComm_t comm);
  std::pair<uint32_t*, uint32_t*> _GetNcclCommAbortFlagPtr(ncclComm_t comm);

  std::string GetDevNcclCommMapKeySetStrs();
  std::string GetDevNcclCommMapKeySetStrsUnlocked() const;

  static ::c10::intrusive_ptr<ProcessGroupNCCL> default_pg_;
  
  // our own maintained abort flag outside of NCCL, 
  // therefore we check abort flag without get the NCCLComm
  std::atomic<uint32_t> abort_flag_{0};
};

}