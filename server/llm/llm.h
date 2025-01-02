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

class LLMWrapper;
class LLMServer {
 public:
  using LLMRequest = std::pair<uint64_t, std::string>;

  static void Init();
  static bp::object GetLLMPyModule() { 
    return llm_server_->py_module_; 
  }
  static bool IsLLMModel(const std::string &model_name);
  static bool AddJob(network::InferHandler::InferData *data);
  std::vector<LLMRequest> GetLLMRequests(
      int batch_size, bool block);

  LLMServer();
  // ~LLMServer();

 private:
  void PyInit();

  static std::unique_ptr<LLMServer> llm_server_;
  
  bp::object py_module_;
  PyThreadState* main_py_ts_;
  BatchJobQueue job_queue_;
  std::unordered_map<uint64_t, std::shared_ptr<Job>> flight_reqs_;
  std::vector<std::unique_ptr<LLMWrapper>> llm_wrappers_;
};

class LLMWrapper {
 public:
  LLMWrapper(int rank, bp::object py_module, pthread_barrier_t* barrier);

  friend LLMServer;
 private:
  void Inference(pthread_barrier_t* barrier);

  int rank_;
  // PyThreadState* py_ts_;
  
  bp::object llm_infer_;
  
  std::unique_ptr<std::thread> thread_;
  
};

}