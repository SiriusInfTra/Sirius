#pragma once

#include <server/grpc/grpc_server.h>
#include <server/schedule/job_queue.h>

#include <boost/python.hpp>
#include <Python.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>

namespace colserve {

namespace bp = boost::python;

struct LLMRequestMetric;
struct LLMRequest;

class LLMWrapper;
class LLMServer {
 public:

  static void Init();
  static void Shutdown();
  static bp::object GetLLMPyModule() { 
    return llm_server_->py_module_; 
  }
  static bool IsLLMModel(const std::string &model_name);
  static bool AddJob(network::InferHandler::InferData *data);
  static std::vector<LLMRequest> GetLLMRequests(
      int batch_size, int timeout_ms, bool block);
  static void FinishLLMRequest(int request_id, std::string output,
                               LLMRequestMetric metric);
  static bool IsRunning() { 
    return llm_server_->running_.load(std::memory_order_acquire); 
  }

  LLMServer();
  // ~LLMServer();

 private:
  void PyInit();
  void AddFlightRequest(uint64_t request_id, 
                        std::shared_ptr<LLMInferJob> job);

  static std::unique_ptr<LLMServer> llm_server_;
  
  std::atomic<bool> running_;
  std::mutex mut_;
  bp::object py_module_;
  PyThreadState* main_py_ts_;
  BatchJobQueue job_queue_;
  std::unordered_map<uint64_t, std::shared_ptr<LLMInferJob>> flight_reqs_;
  std::vector<std::unique_ptr<LLMWrapper>> llm_wrappers_;
};

class LLMWrapper {
 public:
  LLMWrapper(int rank, bp::object py_module, pthread_barrier_t* barrier);
  void Shutdown();

  friend LLMServer;
 private:
  void Inference(pthread_barrier_t* barrier);

  int rank_;
  // PyThreadState* py_ts_;
  
  bp::object llm_infer_;
  
  std::unique_ptr<std::thread> thread_;
  
};

}