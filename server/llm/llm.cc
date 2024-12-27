#include "boost/python/import.hpp"
#include "boost/python/object_fwd.hpp"
#include <memory>
#include <server/llm/llm.h>
#include <string>

namespace colserve {

std::unique_ptr<LLMServer> LLMServer::llm_server_ = nullptr;
bpy::object LLMServer::py_module_;

void LLMServer::Init(
    const std::string &model_name, 
    int max_seq_len, int max_batch_size) {
  Py_Initialize();

  py_module_ = bpy::import("python.llm");


  llm_server_ = std::make_unique<LLMServer>(
      model_name, max_seq_len, max_batch_size);
}

LLMServer::LLMServer(const std::string &model_name, 
                     int max_seq_len, int max_batch_size) {
  llm_infer_ = py_module_.attr("LLMInference")(
      model_name, max_seq_len, max_batch_size);
  
  
}

}

