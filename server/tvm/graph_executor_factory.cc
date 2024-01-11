#include "logging_as_glog.h"
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "graph_executor.h"
#include "graph_executor_factory.h"
#include "texture.h"

#include "../model_infer_store.h"

#include <glog/logging.h>

namespace colserve {
namespace tvm {

GraphExecutorFactory::GraphExecutorFactory(
    size_t rank, ::colserve::Model *model,
    const std::string &model_name,
    const std::string &graph_json,
    const ::tvm::runtime::Module mod,
    const std::string &params_file,
    const std::vector<DLDevice> &devs) : model_rank_(rank), infer_model_(model), model_name_(model_name) {
  std::ifstream graph_json_ifs{graph_json};
  std::string graph_json_str{(std::istreambuf_iterator<char>(graph_json_ifs)),
                             std::istreambuf_iterator<char>()};
  std::istringstream is(graph_json_str);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  for (size_t i = 0; i < nodes_.size(); i++) {
    node_map_[nodes_[i].name] = i;
  }
  module_ = mod;
  devices_ = devs;
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    auto nid = input_nodes_[i];
    std::string &name = nodes_[nid].name;
    input_map_[name] = i;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto nid = outputs_[i].node_id;
    std::string &name = nodes_[nid].name;
    // std::stringstream ss;
    // ss << name << ":" << i;
    output_map_[name] = i;
  }

  // load_param_stream_ = ::tvm::runtime::DeviceAPI::Get(devices_[0])
  //     ->CreateStream(devices_[0]);
  LoadParams(params_file);
  SetupStorage();
}

GraphExecutorFactory::GraphExecutorFactory(
    size_t rank, ::colserve::Model *model,
    const std::string &model_name,
    const std::string &graph_json,
    const ::tvm::runtime::Module mod,
    const std::map<std::string, TVMArray> &params,
    const std::vector<DLDevice> &devs) : model_rank_(rank), infer_model_(model), model_name_(model_name) {
  std::ifstream graph_json_ifs{graph_json};
  std::string graph_json_str{(std::istreambuf_iterator<char>(graph_json_ifs)),
                             std::istreambuf_iterator<char>()};
  std::istringstream is(graph_json_str);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  for (size_t i = 0; i < nodes_.size(); i++) {
    node_map_[nodes_[i].name] = i;
  }
  module_ = mod;
  devices_ = devs;
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    auto nid = input_nodes_[i];
    std::string &name = nodes_[nid].name;
    input_map_[name] = i;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto nid = outputs_[i].node_id;
    std::string &name = nodes_[nid].name;
    // std::stringstream ss;
    // ss << name << ":" << i;
    output_map_[name] = i;
  }

  // load_param_stream_ = ::tvm::runtime::DeviceAPI::Get(devices_[0])
  //     ->CreateStream(devices_[0]);
  LoadParams(params);
  SetupStorage();
}

void GraphExecutorFactory::SetupStorage() {
  std::vector<DLDataType> vtype;
  for (const std::string &s_type : attrs_.dltype) {
    vtype.push_back(::tvm::runtime::String2DLDataType(s_type));
  }

  // std::vector<PoolEntry> pool_entry_;
  for (size_t i = 0; i < attrs_.shape.size(); i++) {
    int storage_id = attrs_.storage_id[i];
    std::string storage_scope = attrs_.storage_scope.empty() ? "" : attrs_.storage_scope[i];
    int device_type = static_cast<int>(devices_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
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
    CheckNullLinkedParam(module_, sid);
    pool_entry_[sid].param_data_entry = i;
    pool_entry_[sid].device_type = device_type;
    pool_entry_[sid].scope = storage_scope;
    if (params_.count(i)) {
      pool_entry_[sid].params_entry = true;
    }

    DLDataType t = vtype[i];
    if (!::tvm::runtime::IsTextureStorage(storage_scope)) {
      size_t size = 1;
      for (int64_t sz : attrs_.shape[i]) {
        size *= static_cast<size_t>(sz);
      }
      size_t bits = t.bits * t.lanes;
      CHECK(bits % 8U == 0U || bits == 1U || bits == 4U);
      int64_t bytes = ((bits + 7U) / 8U) * size;
      pool_entry_[sid].shape[0] = std::max(pool_entry_[sid].shape[0], bytes);
      pool_entry_[sid].dtype = DLDataType{kDLFloat, 32, 1};
    } else {
      CHECK(false) << "texture memory";
      if (pool_entry_[sid].shape.size() == 1) {
        pool_entry_[sid].shape.resize(3, 0);
      }
      size_t axis = ::tvm::runtime::DefaultTextureLayoutSeparator(
          attrs_.shape[i].size(), storage_scope);
      auto shape = ::tvm::runtime::ApplyTexture2DFlattening<int64_t>(
          attrs_.shape[i], attrs_.shape[i].size(), axis);
      pool_entry_[sid].shape[0] = std::max(pool_entry_[sid].shape[0], shape.height);
      pool_entry_[sid].shape[1] = std::max(pool_entry_[sid].shape[1], shape.width);
      CHECK(pool_entry_[sid].shape[2] == 0 || pool_entry_[sid].shape[2] == shape.channel)
          << pool_entry_[sid].shape[2] << " != " << shape.channel
          << ",  texture channel length must be consistent within a storage pool";
      pool_entry_[sid].shape[2] = shape.channel;
      CHECK(pool_entry_[sid].dtype.bits == 0 || ::tvm::runtime::TypeEqual(pool_entry_[sid].dtype, t))
          << ::tvm::runtime::DLDataType2String(pool_entry_[sid].dtype) << " != " << ::tvm::runtime::DLDataType2String(t)
          << ", pool entry for 2d texure allocations must be of the same type;"
          << " downstream error from memory planner likely";
      pool_entry_[sid].dtype = t;
    }
  }

  // for (const auto &pit : pool_entry_) {
  // for (size_t sid = 0; sid < pool_entry_.size(); sid++) {
  //   const auto &pit = pool_entry_[sid];
  //   if (!pit.params_entry) {
  //     continue;
  //   }
  //   const auto &cit = std::find_if(devices_.begin(), devices_.end(), [&pit](const DLDevice &d){
  //     return static_cast<int>(d.device_type) == pit.device_type;
  //   });
  //   DLDevice dev = cit == devices_.end() ? devices_[0] : *cit;
  //   std::vector<int64_t> shape = pit.shape;
  //   if (shape.size() == 1) {
  //     shape[0] = (shape[0] + 3) / 4;
  //   }
  //   ::tvm::runtime::Optional<::tvm::runtime::String> mem_scope;
  //   if (!pit.scope.empty()) {
  //     mem_scope = ::tvm::runtime::String(pit.scope);
  //   }
  //   storage_pool_.push_back(TVMArray::Empty(shape, pit.dtype, dev, mem_scope));
  //   param_node_storage_id_map_[sid] = storage_pool_.size() - 1;
  // }

  // for (auto &p : params_) {
  //   auto storage_id = attrs_.storage_id[p.first];
  //   storage_pool_[param_node_storage_id_map_[storage_id]].CopyFrom(p.second);
  // }
}

void GraphExecutorFactory::LoadParams(const std::string &params_file) {
  struct stat file_stat;
  int params_fd = open(params_file.c_str(), O_RDONLY);
  fstat(params_fd, &file_stat);
  auto params_ptr = static_cast<const char*>(
      mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, params_fd, 0U));
  std::string params_blob;
  params_blob.reserve(file_stat.st_size);
  params_blob.assign(params_ptr, params_ptr + file_stat.st_size);
  munmap((void*)params_ptr, file_stat.st_size);
  // std::ifstream ifs{params_file, std::ios::binary};
  // std::string params_blob{(std::istreambuf_iterator<char>(ifs)),
  //                          std::istreambuf_iterator<char>()};
  dmlc::MemoryStringStream ms_strm(const_cast<std::string*>(&params_blob));
  dmlc::Stream* strm = &ms_strm;
  uint64_t header, reserved;
  constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;
  CHECK(strm->Read(&header)) << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic) << "Invalid parameters file format";
  CHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size()) << "Invalid parameters file format";
  size_t params_size = 0;
  for (size_t i = 0; i < size; ++i) {
    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    TVMArray temp;
    temp.Load(strm);

    CHECK(input_map_.count(names[i])) << "cannot find " << names[i] <<  " in model parameter"; 
    auto it = node_map_.find(names[i]);
    // LOG(INFO) << "find param in input_map " << it->first << " " << it->second;

    params_[it->second] = temp.CopyTo(::tvm::Device{kDLCUDAHost, 0});
    params_size += ::tvm::runtime::GetDataSize(*params_[it->second].operator->());
  }
  VLOG(1) << params_file << " " << 1.0 * params_size / 1024 / 1024 << " Mb";
}

void GraphExecutorFactory::LoadParams(const std::map<std::string, TVMArray> &params) {
  for (const auto &p : params) {
    CHECK(input_map_.count(p.first)) << "cannot find " << p.first <<  " in model parameter";
    auto it = node_map_.find(p.first);
    params_[it->second] = p.second.CopyTo(::tvm::Device{kDLCUDAHost, 0});
  }
}

std::map<std::string, TVMArray> GraphExecutorFactory::LoadParamsAsTVMArray(const std::string &params_file) {
  struct stat file_stat;
  int params_fd = open(params_file.c_str(), O_RDONLY);
  fstat(params_fd, &file_stat);
  auto params_ptr = static_cast<const char*>(
      mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, params_fd, 0U));
  std::string params_blob;
  params_blob.reserve(file_stat.st_size);
  params_blob.assign(params_ptr, params_ptr + file_stat.st_size);
  munmap((void*)params_ptr, file_stat.st_size);
  // std::ifstream ifs{params_file, std::ios::binary};
  // std::string params_blob{(std::istreambuf_iterator<char>(ifs)),
  //                          std::istreambuf_iterator<char>()};
  dmlc::MemoryStringStream ms_strm(const_cast<std::string*>(&params_blob));
  dmlc::Stream* strm = &ms_strm;
  uint64_t header, reserved;
  constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;
  CHECK(strm->Read(&header)) << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic) << "Invalid parameters file format";
  CHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size()) << "Invalid parameters file format";
  size_t params_size = 0;
  std::map<std::string, TVMArray> params_ret;
  for (size_t i = 0; i < size; ++i) {
    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    TVMArray temp;
    temp.Load(strm);

    params_ret[names[i]] = temp;
    params_size += ::tvm::runtime::GetDataSize(*temp.operator->());

    // CHECK(input_map_.count(names[i])) << "cannot find " << names[i] <<  " in model parameter"; 
    // auto it = node_map_.find(names[i]);
    // LOG(INFO) << "find param in input_map " << it->first << " " << it->second;

    // params_[it->second] = temp.CopyTo(::tvm::Device{kDLCUDAHost, 0});
    // params_size += ::tvm::runtime::GetDataSize(*params_[it->second].operator->());
  }
  VLOG(1) << params_file << " " << 1.0 * params_size / 1024 / 1024 << " Mb";
  return params_ret;
}

std::unique_ptr<GraphExecutor> GraphExecutorFactory::CreateGraphExecutor(size_t worker_id) {
  return std::make_unique<GraphExecutor>(*this, worker_id);
}

std::tuple<GraphExecutorFactory::ShapeInfo, GraphExecutorFactory::DtypeInfo>
    GraphExecutorFactory::GetInputInfo() const {
  ShapeInfo shape_info;
  DtypeInfo dtype_info;
  for (auto nid : input_nodes_) {
    if (!params_.count(nid)) {
      shape_info[nodes_[nid].name] = attrs_.shape[nid];
      dtype_info[nodes_[nid].name] = attrs_.dltype[nid];
    }
  }
  return {shape_info, dtype_info};
}

std::tuple<GraphExecutorFactory::ShapeInfo, GraphExecutorFactory::DtypeInfo>
    GraphExecutorFactory::GetOutputInfo() const {
  ShapeInfo shape_info;
  DtypeInfo dtype_info;
  for (auto e : outputs_) {
    auto nid = e.node_id;
    shape_info[nodes_[nid].name] = attrs_.shape[nid];
    dtype_info[nodes_[nid].name] = attrs_.dltype[nid];
  }
  return {shape_info, dtype_info};
}

// void GraphExecutorFactory::ResetParamStorage() {
//   using namespace ::tvm::runtime;
//   for (auto &it : param_ready_) {
//     it.second = false;
//   }
//   for (auto &s : storage_pool_) {
//     DeviceAPI::Get(s->device)->FreeDataSpace(s->device, s->data);
//     s.get_mutable()->dl_tensor.data = nullptr;
//   }
// }

// void GraphExecutorFactory::AllocParamStorage() {
//   using namespace ::tvm::runtime;
//   for (auto &s : storage_pool_) {
//     s.get_mutable()->dl_tensor.data =
//         DeviceAPI::Get(s->device)->AllocDataSpace(
//           s->device, s->ndim, s->shape, s->dtype);
//   }
  // for (auto &p : params_) {
  //   auto sid = attrs_.storage_id[p.first];
  //   auto &pit = pool_entry_[sid];
  //   CHECK(pit.params_entry);
  //   auto &storage = storage_pool_[param_node_storage_id_map_[sid]];
  //   Optional<String> mem_scope;
  //   if (!pit.scope.empty()) {
  //     mem_scope = String(pit.scope);
  //   }
  //   storage.get_mutable()->dl_tensor.data =
  //       DeviceAPI::Get(storage->device)->AllocDataSpace(
  //         storage->device, storage->ndim, storage->shape, storage->dtype, mem_scope);
  // }
// }

// void GraphExecutorFactory::PipelineLoadParams() {
//   for (auto &p : params_) {
//     auto sid = attrs_.storage_id[p.first];
//     sid = param_node_storage_id_map_[sid];
//     if (!param_ready_[p.first]) {
//       tvm::TVMArray::CopyFromTo(
//         p.second.operator->(), &storage_pool_[sid].get_mutable()->dl_tensor, load_param_stream_);
//       ::tvm::runtime::DeviceAPI::Get(storage_pool_[sid]->device)
//           ->StreamSync(storage_pool_[sid]->device, load_param_stream_);
//       param_ready_[p.first] = true;
//     }
//   }
// }

}
}