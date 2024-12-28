#pragma once

#include <Python.h>
#include <boost/python.hpp>
#include <memory>
#include <string>
#include <vector>

namespace colserve {

namespace bp = boost::python;

class LLMWrapper;
class LLMServer {
 public:
  static void Init();

  LLMServer();
  // ~LLMServer();

 private:
  static std::unique_ptr<LLMServer> llm_server_;
  
  std::vector<std::unique_ptr<LLMWrapper>> llm_wrappers_;
};

class LLMWrapper {
 public:
  LLMWrapper();

  friend LLMServer;
 private:
  static bp::object py_module_;
  bp::object llm_infer_;

};

}