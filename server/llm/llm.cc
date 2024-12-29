#include <server/logging_as_glog.h>
#include <server/config.h>
#include <server/llm/llm.h>
#include <server/llm/torch_allocator_plugin.h>

#include <boost/format.hpp>
#include <boost/range/irange.hpp>
#include <filesystem>

#include <memory>
#include <string>

namespace colserve {

std::unique_ptr<LLMServer> LLMServer::llm_server_ = nullptr;

void LLMServer::Init() {
  try {
    Py_Initialize();
    LOG(INFO) << "[LLMServer] python initialized";
    llm_server_ = std::make_unique<LLMServer>();
  } catch (boost::python::error_already_set const &) {
    PyErr_Print();
    LOG(ERROR) << "Error importing python module";
  }
  LOG(INFO) << "[LLMServer] initialized";
}

LLMServer::LLMServer() {
  PyInit();

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, 1 + llm_wrappers_.size());
  for (auto i : boost::irange(sta::DeviceManager::GetNumVisibleGpu())) {
    llm_wrappers_.push_back(std::make_unique<LLMWrapper>(
        i, py_module_, &barrier));
  }
  pthread_barrier_wait(&barrier);
}

void LLMServer::PyInit() {
  try {
    main_py_ts_ = PyThreadState_GET();
    // add the current directory to the python path
    bp::exec((boost::format(
      "import sys\nsys.path.append(%s)") % 
      (std::filesystem::path(__FILE__).parent_path() / "python")
    ).str().c_str());
    bp::exec((boost::format(
      "import sys\nprint(sys.path)\n")
    ).str().c_str());

    py_module_ = bp::import("llm");

    if (Config::use_shared_tensor_infer) {
      torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
      torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()
          ->init(sta::DeviceManager::GetNumVisibleGpu());
      torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()
          ->SetCurrentAllocator();
    }
  } catch (const bp::error_already_set&) {
    PyErr_Print();
    LOG(FATAL) << "[LLM Server] python module init failed";
  }
}

LLMWrapper::LLMWrapper(int rank, 
                       bp::object py_module, 
                       pthread_barrier_t* barrier) 
    : rank_(rank) {
  try {
    llm_infer_ =  py_module.attr("LLMInference")(
        Config::llm_model_name,
        Config::llm_max_seq_len,
        Config::llm_max_batch_size
    );
  } catch (const bp::error_already_set&) {
    PyErr_Print();
    LOG(FATAL) << "[LLMWrapper] create infer backend failed";
  }
  thread_.reset(new std::thread(&LLMWrapper::Inference, this, barrier));
}

void LLMWrapper::Inference(pthread_barrier_t* barrier) {
  pthread_barrier_wait(barrier);
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

} // namespace colserve

