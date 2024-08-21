#ifndef COLSERVE_GRAPH_EXECUTOR_FACTORY_H
#define COLSERVE_GRAPH_EXECUTOR_FACTORY_H
#include <server/logging_as_glog.h>
#include <server/config.h>

#include <common/util.h>
#include <common/tensor/tensor.h>

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>

#include <fstream>
#include <iostream>
#include <map>
#include <future>
#include <string>

// #include "tvm.h"


namespace colserve {

class Model;
std::string GetModelNameWithoutDuplicatedId(
    const std::string &model_name);

namespace tvm {

// TODO read from configuration

template<size_t align>
inline size_t AlignedNBytes(size_t nbytes) {
  static_assert((align & (align - 1)) == 0, 
                "alignment must be power of 2");
  return (nbytes + (align - 1)) & (~(align - 1));
}

static const constexpr size_t MEM_BLOCK_NBYTES = 32_MB;
static const constexpr size_t MEM_BLOCK_ALIGN_NBYTES = 16_MB;
inline size_t GetMemBlockAlignedNBytes(size_t nbytes) {
  return nbytes >= MEM_BLOCK_NBYTES 
         ? AlignedNBytes<MEM_BLOCK_NBYTES>(nbytes) 
         : AlignedNBytes<MEM_BLOCK_ALIGN_NBYTES>(nbytes);
}

constexpr size_t LINE_ALIGN_NBYTES = 128_B;
inline size_t GetLineAlignedNbytes(size_t nbytes) {
  return AlignedNBytes<LINE_ALIGN_NBYTES>(nbytes);
}

using TVMArray = ::tvm::runtime::NDArray;

/*! \brief operator attributes about tvm op */
struct TVMOpParam {
  std::string func_name;
  std::unordered_map<std::string, std::string> attrs;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
};

struct OpArgs {
  std::vector<DLTensor*> args;
  std::vector<TVMValue> arg_values;
  std::vector<int> arg_tcodes;
  std::vector<int64_t> shape_data;
};

struct PoolEntry {
  memory_byte_t nbytes;
  bool is_param_entry;
};

struct NodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  inline bool operator==(const NodeEntry& other) const {
    return node_id == other.node_id 
        && index == other.index 
        && version == other.version;
  }
  // JSON Loader
  void Load(dmlc::JSONReader* reader) {
    reader->BeginArray();
    ICHECK(reader->NextArrayItem()) << "invalid json format";
    reader->Read(&node_id);
    ICHECK(reader->NextArrayItem()) << "invalid json format";
    reader->Read(&index);
    if (reader->NextArrayItem()) {
      reader->Read(&version);
      ICHECK(!reader->NextArrayItem()) << "invalid json format";
    } else {
      version = 0;
    }
  }
};

struct Node {
  // operator type in string
  std::string op_type;
  // name of the op
  std::string name;
  // parameters
  TVMOpParam param;
  // inputs
  std::vector<NodeEntry> inputs;
  // control deps
  std::vector<uint32_t> control_deps;

  // JSON Loader
  void LoadAttrs(dmlc::JSONReader* reader, TVMOpParam* param) {
    int bitmask = 0;
    std::string key, value;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      reader->Read(&value);
      if (key == "func_name") {
        param->func_name = value;
        bitmask |= 1;
      } else if (key == "num_inputs") {
        param->num_inputs = strtoul(value.c_str(), nullptr, 10);
        bitmask |= 2;
      } else if (key == "num_outputs") {
        param->num_outputs = strtoul(value.c_str(), nullptr, 10);
        bitmask |= 4;
      } else if (key == "flatten_data") {
        param->flatten_data = strtoul(value.c_str(), nullptr, 10);
        bitmask |= 8;
      } else {
        param->attrs[key] = std::string(value);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "invalid format";
  }
  // JSON Loader
  void Load(dmlc::JSONReader* reader) {
    reader->BeginObject();
    int bitmask = 0;
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "op") {
        reader->Read(&op_type);
        bitmask |= 1;
      } else if (key == "name") {
        reader->Read(&name);
        bitmask |= 2;
      } else if (key == "inputs") {
        reader->Read(&inputs);
        bitmask |= 4;
      } else if (key == "attr" || key == "attrs") {
        this->LoadAttrs(reader, &param);
      } else if (key == "control_deps") {
        reader->Read(&control_deps);
      } else {
        ICHECK(false) << "do not support key " << key;
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4) << "invalid format";
  }
};

struct GraphAttr {
  size_t storage_num_not_alloctaed{0};
  std::vector<int> storage_id;
  std::vector<int> device_index;
  std::vector<std::string> dltype;
  std::vector<std::string> storage_scope;
  std::vector<std::vector<int64_t>> shape;
  // The graph attribute fields.
  void Load(dmlc::JSONReader* reader) {
    reader->BeginObject();
    int bitmask = 0;
    std::string key, type;
    while (reader->NextObjectItem(&key)) {
      if (key == "dltype") {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&type);
        ICHECK_EQ(type, "list_str");
        ICHECK(reader->NextArrayItem());
        reader->Read(&dltype);
        ICHECK(!reader->NextArrayItem());
        bitmask |= 1;
      } else if (key == "storage_id") {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&type);
        ICHECK_EQ(type, "list_int");
        ICHECK(reader->NextArrayItem());
        reader->Read(&storage_id);
        ICHECK(!reader->NextArrayItem());
        bitmask |= 2;
      } else if (key == "storage_scope") {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&type);
        ICHECK_EQ(type, "list_str");
        ICHECK(reader->NextArrayItem());
        reader->Read(&storage_scope);
        ICHECK(!reader->NextArrayItem());
        bitmask |= 1;
      } else if (key == "shape") {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&type);
        ICHECK_EQ(type, "list_shape");
        ICHECK(reader->NextArrayItem());
        reader->Read(&shape);
        ICHECK(!reader->NextArrayItem());
        bitmask |= 4;
      } else if (key == "device_index") {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&type);
        ICHECK_EQ(type, "list_int");
        ICHECK(reader->NextArrayItem());
        reader->Read(&device_index);
        ICHECK(!reader->NextArrayItem());
      } else {
        reader->BeginArray();
        ICHECK(reader->NextArrayItem());
        reader->Read(&type);
        if (type == "list_int") {
          ICHECK(reader->NextArrayItem());
          std::vector<int> temp;
          reader->Read(&temp);
        } else if (type == "size_t") {
          ICHECK(reader->NextArrayItem());
          size_t temp;
          reader->Read(&temp);
        } else {
          ICHECK(false) << "cannot skip graph attr " << key;
        }
        ICHECK(!reader->NextArrayItem());
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4) << "invalid format";
  }
};


class Executor;
class TVMGraph {
 public:
  static std::string mod_json;
  static std::string mod_so;
  static std::string mod_params;
  static std::string mod_group;

  static std::map<std::string, TVMArray> 
      LoadParamsAsTVMArray(const std::string &params_file);


  TVMGraph(size_t rank, ::colserve::Model* infer_model,
                       const std::string &model_name,
                       const std::filesystem::path &model_path,
                       const std::string &graph_json,
                       const std::string &group_txt,
                       const ::tvm::runtime::Module mod,
                       const std::string &params_file);
  TVMGraph(size_t rank, ::colserve::Model* infer_model,
                       const std::string &model_name,
                       const std::filesystem::path &model_path,
                       const std::string &graph_json,
                       const std::string &group_txt,
                       const ::tvm::runtime::Module mod,
                       const std::map<std::string, TVMArray> &params);
  std::unique_ptr<Executor> CreateGraphExecutor(size_t worker_id, const std::vector<DLDevice> &devs);

  using ShapeInfo = std::map<std::string, std::vector<int64_t>>;
  using DtypeInfo = std::map<std::string, std::string>;
  std::tuple<ShapeInfo, DtypeInfo> GetInputInfo() const;
  std::tuple<ShapeInfo, DtypeInfo> GetOutputInfo() const;

  memory_byte_t GetParamStorageNBytes() const {
    return param_storage_nbytes_;
  }

  memory_byte_t GetBufferStorageNBytes() const {
    return buffer_storage_nbytes_;
  }

  memory_byte_t GetStorageNBytes() const {
    return param_storage_nbytes_ + buffer_storage_nbytes_;
  }

  std::vector<memory_byte_t> GetGroupsNbytes() const {
    return storage_group_nbytes_;
  }

  memory_byte_t GetStorageAlignedNBytes() const {
    if (Config::group_param_load) {
      if (Config::group_param_nbytes_with_fragment) {
        return model_nbytes_with_group_fragment_;
      } else {
        return AlignedNBytes<MEM_BLOCK_ALIGN_NBYTES>(GetStorageNBytes());
      }
    } else {
      return GetStorageNBytes();
    }
  }

  inline size_t GetModelRank() const { return model_rank_; }

  uint32_t entry_id(NodeEntry e) const {
    return node_row_ptr_[e.node_id] + e.index;
  }
  uint32_t entry_id(uint32_t nid, uint32_t index) const {
    return node_row_ptr_[nid] + index;
  }

  // void ResetParamStorage();
  // void AllocParamStorage();
  // void PipelineLoadParams();

  friend class Executor;
 private:
  // void SetupStorage();
  void LoadGraph(const std::string &graph_json);
  void LoadParams(const std::string &params_file);
  void LoadParams(const std::map<std::string, TVMArray> &params);
  void SetupStorage();
  void SetupHostPinnedIOStorage();
  void SetupStorageGroup();
  void SetupParamGroupPartition(const std::string &path);

  void DumpParamGroupPartition();

  void Load(dmlc::JSONReader* reader) {
    reader->BeginObject();
    int bitmask = 0;
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "nodes") {
        reader->Read(&nodes_);
        bitmask |= 1;
      } else if (key == "arg_nodes") {
        reader->Read(&input_nodes_);
        bitmask |= 2;
      } else if (key == "node_row_ptr") {
        reader->Read(&node_row_ptr_);
        bitmask |= 4;
      } else if (key == "heads") {
        reader->Read(&outputs_);
        bitmask |= 8;
      } else if (key == "attrs") {
        reader->Read(&attrs_);
        bitmask |= 16;
      } else if (key == "metadata") {
        break;
      } else {
        ICHECK(false) << "key " << key << " is not supported";
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8 | 16) << "invalid format";
  }
  bool CheckNullLinkedParam(::tvm::runtime::Module mod, int64_t storage_id) {
    auto module_lookup_linked_param =
        mod.GetFunction(::tvm::runtime::symbol::tvm_lookup_linked_param, true);
    if (module_lookup_linked_param == nullptr) {
      return true;
    }
    auto opaque_handle = module_lookup_linked_param(storage_id);
    if (opaque_handle.type_code() == kTVMNullptr) {
      return true;
    }
    return false;
  }

  size_t model_rank_;
  std::filesystem::path model_path_;
  std::string model_name_;
  ::colserve::Model *infer_model_;
  ::tvm::runtime::Module module_;
  // std::vector<DLDevice> devices_;

  // graph
  std::vector<Node> nodes_;
  std::vector<uint32_t> input_nodes_;
  std::vector<uint32_t> node_row_ptr_;
  std::vector<NodeEntry> outputs_;
  GraphAttr attrs_;

  std::map<std::string, uint32_t> node_map_;
  std::map<std::string, uint32_t> input_map_;
  std::map<std::string, uint32_t> output_map_;

  // data entry id -> TVMArray
  std::map<uint32_t, TVMArray> host_params_; 
  // [ param storage group, [param ids ...] ]
  std::vector<std::pair<sta::STensor, std::vector<uint32_t>>> 
      host_param_storage_group_;

  
  // std::vector<size_t> storage_group_parti_;
  
  // std::map<uint32_t, uint32_t> param_node_storage_id_map_;


  // to avoid alloc pin memory during set input/get output
  // std::unordered_map<std::string, TVMArray> 
  //     input_cpu_pin_bufs_, output_cpu_pin_bufs_;

  // storage meta data
  std::vector<PoolEntry> storage_pool_entries_;
  // ensure param are allocated together if group param loading is enabled
  std::vector<uint32_t> storage_alloc_order_;
  // storage allocation group
  std::vector<size_t> storage_group_partition_;
  std::vector<std::vector<size_t>> storage_group_offsets_;
  std::vector<memory_byte_t> storage_group_nbytes_;
  memory_byte_t param_storage_nbytes_,
                buffer_storage_nbytes_,
                model_nbytes_with_group_fragment_;

  
};

} 
}


#endif