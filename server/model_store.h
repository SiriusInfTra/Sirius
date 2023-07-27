#ifndef COLSERVE_MODEL_STORE_H
#define COLSERVE_MODEL_STORE_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>

#ifdef GLOG_LOGGING_H
  static_assert(false, "glog/glog.h should be included after c.h");
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>

#include "job_queue.h"
#include "grpc/grcp_server.h"

namespace colserve {

class Model;

class ModelStore {
 public:
  static ModelStore* Get();
  static void Init(const std::filesystem::path &model_store_path);

  Model* GetModel(const std::string &name);

 private:
  static std::unique_ptr<ModelStore> model_store_;

  std::unordered_map<std::string, std::unique_ptr<Model>> models_;
};

class Model {
 public:
  Model() : name_("dummy") {};
  Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device);
  bool AddJob(network::InferHandler::InferData* data);

 private:
  void InitMetaInfo();
  bool Inference();
  bool SetInput(const std::string &input_id, const std::vector<std::shared_ptr<Job>> &jobs);
  bool GetOutput(const std::string &output_id, const std::vector<std::shared_ptr<Job>> &jobs);
  
  std::string name_;
  DLDevice device_;
  BatchJobQueue job_queue_;

  tvm::runtime::Module rmod_;
  tvm::runtime::Module graph_executor_;

  tvm::runtime::PackedFunc set_input_;
  tvm::runtime::PackedFunc get_input_;
  tvm::runtime::PackedFunc run_;
  tvm::runtime::PackedFunc get_output_;

  // param_name -> [[shape], dtype]
  std::unordered_map<std::string, 
      std::pair<tvm::runtime::ShapeTuple, std::string>> input_info_, output_info_;

  std::unique_ptr<std::thread> thread_;
};


} // namespace colserve

#endif

