#ifndef COLSERVE_GRAPH_EXECUTOR_FACTORY_H
#define COLSERVE_GRAPH_EXECUTOR_FACTORY_H
#include "../logging_as_glog.h"
#include <iostream>
#include <map>


#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>



// #include "tvm.h"


namespace colserve {

class Model;

namespace tvm {

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
  int device_type;
  std::vector<int64_t> shape;
  DLDataType dtype;
  int param_data_entry;
  TVMArray linked_param;
  std::string scope;
  bool params_entry;
  //    PoolEntry(int s, int dev_type, void* pre_linked_param) :
  //        size(s), device_type(dev_type), pre_linked_param(std::move(pre_linked_param)) {}
};
struct NodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  inline bool operator==(const NodeEntry& other) const {
    return node_id == other.node_id && index == other.index && version == other.version;
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


class GraphExecutor;
class GraphExecutorFactory {
 public:
  GraphExecutorFactory(size_t rank, ::colserve::Model* infer_model,
                       const std::string &model_name,
                       const std::string &graph_json,
                       const ::tvm::runtime::Module mod,
                       const std::string &params_file,
                       const std::vector<DLDevice> &devs);
  GraphExecutorFactory(size_t rank, ::colserve::Model* infer_model,
                       const std::string &model_name,
                       const std::string &graph_json,
                       const ::tvm::runtime::Module mod,
                       const std::map<std::string, TVMArray> &params,
                       const std::vector<DLDevice> &devs);
  std::unique_ptr<GraphExecutor> CreateGraphExecutor(size_t worker_id);

  using ShapeInfo = std::map<std::string, std::vector<int64_t>>;
  using DtypeInfo = std::map<std::string, std::string>;
  std::tuple<ShapeInfo, DtypeInfo> GetInputInfo() const;
  std::tuple<ShapeInfo, DtypeInfo> GetOutputInfo() const;

  inline size_t GetModelRank() const { return model_rank_; }

  static std::map<std::string, TVMArray> LoadParamsAsTVMArray(const std::string &params_file);

  // void ResetParamStorage();
  // void AllocParamStorage();
  // void PipelineLoadParams();

  friend class GraphExecutor;
 private:
  void SetupStorage();
  void LoadParams(const std::string &params_file);
  void LoadParams(const std::map<std::string, TVMArray> &params);
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
  ::colserve::Model *infer_model_;
  std::string model_name_;
  ::tvm::runtime::Module module_;
  std::vector<DLDevice> devices_;

  // graph
  std::vector<Node> nodes_;
  std::vector<uint32_t> input_nodes_;
  std::vector<uint32_t> node_row_ptr_;
  std::vector<NodeEntry> outputs_;
  GraphAttr attrs_;

  std::map<std::string, uint32_t> node_map_;
  std::map<std::string, uint32_t> input_map_;
  std::map<std::string, uint32_t> output_map_;
  std::map<uint32_t, TVMArray> params_;
  // std::map<uint32_t, uint32_t> param_node_storage_id_map_;

  std::vector<PoolEntry> pool_entry_;
  // std::vector<TVMArray> storage_pool_;

};

} 
}


#endif