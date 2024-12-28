#include <server/logging_as_glog.h>
#include <server/config.h>
#include <server/llm/llm.h>

#include <boost/format.hpp>
#include <filesystem>

#include <memory>
#include <string>

namespace colserve {

std::unique_ptr<LLMServer> LLMServer::llm_server_ = nullptr;
bp::object LLMWrapper::py_module_;

void LLMServer::Init() {
  try {
    Py_Initialize();
    LOG(INFO) << "[LLMServer] Initializing python";
    // add the current directory to the python path
    bp::exec((boost::format(
      "import sys\nsys.path.append(%s)") % 
      (std::filesystem::path(__FILE__).parent_path() / "python")
    ).str().c_str());
    bp::exec("import sys\nprint(sys.path)");

    LLMWrapper::py_module_ = bp::import("llm");
    llm_server_ = std::make_unique<LLMServer>();
  } catch (boost::python::error_already_set const &) {
    PyErr_Print();
    LOG(ERROR) << "Error importing python module";
  }

  LOG(INFO) << "[LLMServer] initialized";
}

LLMServer::LLMServer() {
  llm_wrappers_.push_back(std::make_unique<LLMWrapper>());

  // llm_infer_ = py_module_.attr("LLMInference")(
  //   COnfi
  // );
}

LLMWrapper::LLMWrapper() {
  llm_infer_ = py_module_.attr("LLMInference")(
      Config::llm_model_name,
      Config::llm_max_seq_len,
      Config::llm_max_batch_size);
  
}

}

