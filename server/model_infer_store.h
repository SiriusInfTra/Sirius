#ifndef COLSERVE_MODEL_INFER_STORE_H
#define COLSERVE_MODEL_INFER_STORE_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>

#ifdef GLOG_LOGGING_H
  static_assert(false, "glog/glog.h should be included after this file");
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include "tvm/graph_executor.h"
#include "job_queue.h"
#include "grpc/grcp_server.h"

namespace colserve {

class Model;

class ModelInferStore {
 public:
  static ModelInferStore* Get();
  static void Init(const std::filesystem::path &infer_store_path);
  static bool Shutdown() { return true; }


  Model* GetModel(const std::string &name);
  size_t NumJobs();

 private:
  static std::unique_ptr<ModelInferStore> model_infer_store_;

  std::unordered_map<std::string, std::unique_ptr<Model>> models_;
};

class Model {
 public:
  Model() : name_("dummy") {};
  Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device, 
        size_t batch_size, size_t num_worker = 1);
  bool AddJob(network::InferHandler::InferData* data);
  size_t NumJobs() { return job_queue_.NumJobs(); }

 private:
  void InitMetaInfo();
  bool Inference(pthread_barrier_t* barrier = nullptr);
  bool SetInput(tvm::GraphExecutor &graph_executor, size_t idx, const std::string &input_id, 
                const std::vector<std::shared_ptr<Job>> &jobs);
  bool GetOutput(tvm::GraphExecutor &graph_executor, 
                 size_t idx, const std::string &output_id, const std::vector<std::shared_ptr<Job>> &jobs);
  void MonitorJob();
  
  std::string name_;
  DLDevice device_;
  size_t batch_size_;
  BatchJobQueue job_queue_;

  std::unique_ptr<tvm::GraphExecutorFactory> graph_executor_factory_;

  // infer scaling
  double scale_up_queue_time_;
  double scale_down_idle_time_;

  // param_name -> [[shape], dtype]
  std::unordered_map<std::string, 
      std::pair<std::vector<int64_t>, std::string>> input_info_, output_info_;

  // std::unique_ptr<std::thread> thread_;
  std::vector<std::unique_ptr<std::thread>> infer_workers_;
  std::unique_ptr<std::thread> job_monitor_;
};


} // namespace colserve

#endif

