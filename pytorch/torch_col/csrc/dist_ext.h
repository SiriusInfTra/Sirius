#pragma once

#include <torch/csrc/distributed/c10d/reducer.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

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


  std::vector<ncclComm_t> GetNcclComm(
      const std::vector<at::Device> &device_key) const;

  void RestartNcclComm(
      const std::vector<at::Device> &devices);

 private:
  // from torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp,
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
};

}