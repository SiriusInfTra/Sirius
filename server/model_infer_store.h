#ifndef COLSERVE_MODEL_INFER_STORE_H
#define COLSERVE_MODEL_INFER_STORE_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <optional>
#include <cstdlib>


// #include <dlpack/dlpack.h>
// #include <tvm/runtime/module.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/registry.h>
// #include <tvm/runtime/c_runtime_api.h>
// #include <tvm/runtime/device_api.h>

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

  static std::atomic<size_t> model_rank;

  Model* GetModel(const std::string &name);
  size_t NumJobs();

 private:
  static std::unique_ptr<ModelInferStore> model_infer_store_;

  std::unordered_map<std::string, std::unique_ptr<Model>> models_;
};

class Model {
 public:
  Model() : name_("dummy") {};
  // Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device, 
  //       size_t batch_size, size_t num_worker, size_t max_num_worker);
  Model(const std::string &name, const std::filesystem::path &model_path,
        std::optional<const std::map<std::string, tvm::TVMArray>> params,
        DLDevice device, size_t batch_size, size_t num_worker, size_t max_num_worker);
  bool AddJob(network::InferHandler::InferData* data);
  size_t NumJobs() { return job_queue_.NumJobs(); }

  void SetWaitTrainPid(size_t worker_id, pid_t train_pid) {
    CHECK_LT(worker_id, waited_trains_.size());
    waited_trains_[worker_id] = train_pid;
  }

 private:
  void InitMetaInfo();
  bool Inference(uint32_t rank, pthread_barrier_t* barrier);
  bool SetInput(tvm::GraphExecutor &graph_executor, size_t idx, const std::string &input_id, 
                const std::vector<std::shared_ptr<Job>> &jobs);
  bool GetOutput(tvm::GraphExecutor &graph_executor, 
                 size_t idx, const std::string &output_id, const std::vector<std::shared_ptr<Job>> &jobs);
  void MonitorJob();
  
  std::string name_;
  DLDevice device_;
  size_t batch_size_;
  BatchJobQueue job_queue_;

  std::vector<std::unique_ptr<tvm::GraphExecutor>> graph_executor_pool_;
  std::unique_ptr<tvm::GraphExecutorFactory> graph_executor_factory_;

  // infer scaling
  double scale_up_queue_time_;  // ms
  double scale_down_idle_time_; // ms
  uint32_t max_num_worker_;
  std::atomic<uint32_t> num_worker_;
  std::vector<std::unique_ptr<std::atomic<bool>>> worker_running_;

  std::vector<pid_t> waited_trains_; 

  // param_name -> [[shape], dtype]
  std::unordered_map<std::string, 
      std::pair<std::vector<int64_t>, std::string>> input_info_, output_info_;

  // std::unique_ptr<std::thread> thread_;
  std::vector<std::unique_ptr<std::thread>> infer_workers_;
  std::unique_ptr<std::thread> job_monitor_;
  
};


} // namespace colserve

#endif

