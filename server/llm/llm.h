#pragma once

#include <memory>
#include <Python.h>
#include <boost/python.hpp>
#include <object.h>
#include <string>

namespace colserve {

namespace bpy = boost::python;

class LLMServer {
 public:
  static void Init(const std::string &model_name,
                   int max_seq_len, int max_batch_size);

  LLMServer(const std::string &model_name,
            int max_seq_len, int max_batch_size);
  // ~LLMServer();

 private:
  static std::unique_ptr<LLMServer> llm_server_;
  static bpy::object py_module_;

  bpy::object llm_infer_;
};

}