#include <server/logging_as_glog.h>
#include <server/config.h>
#include <server/llm/llm.h>
#include <server/llm/llm_util.h>
#include <server/llm/torch_allocator_plugin.h>

#include <boost/format.hpp>
#include <boost/range/irange.hpp>
#include <filesystem>

#include <memory>
#include <string>

namespace colserve {

std::unique_ptr<LLMServer> LLMServer::llm_server_ = nullptr;

BOOST_PYTHON_MODULE(llm_server)
{
  bp::def("get_llm_requests", &LLMServer::GetLLMRequests);
  bp::def("info", &CallGLOG_INFO);
  bp::def("dinfo", &CallGLOG_DINFO);
}

void LLMServer::Init() {
  try {
    PyImport_AppendInittab("llm_server", &PyInit_llm_server);
    Py_Initialize();
    LOG(INFO) << "[LLMServer] python initialized";
    llm_server_ = std::make_unique<LLMServer>();
  } catch (boost::python::error_already_set const &) {
    PyErr_Print();
    LOG(ERROR) << "Error importing python module";
  }
  LOG(INFO) << "[LLMServer] initialized";

}

bool LLMServer::IsLLMModel(const std::string &model_name) {
  if (model_name.find("llama") != std::string::npos) {
    return true;
  }
  return false;
}

bool LLMServer::AddJob(network::InferHandler::InferData *data) {
  if (data->GetModelName() != Config::llm_model_name) {
    return false;
  }

  CHECK(llm_server_ != nullptr);
  CHECK(llm_server_->llm_wrappers_.size() > 0);
  // llm_server_->llm_wrappers_.front()->AddJob(data);
  llm_server_->job_queue_.Put(std::make_shared<InferJob>(data));
  return true;
}

std::vector<LLMServer::LLMRequest>
LLMServer::GetLLMRequests(int batch_size, bool block) {
  while (true) {
    auto jobs = job_queue_.GetBatch(batch_size, 10, 10);
    if (jobs.size() > 0) {
      std::vector<LLMRequest> llm_reqs;
      for (auto & job : jobs) {
        auto req_id = job->GetInferData()->GetId();
        flight_reqs_[req_id] = job;
        llm_reqs.push_back({req_id, job->GetInferData()->GetInputData(0)});
      }
      return llm_reqs;
    } else if (!block) {
      return {};
    }
  }
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

    bp::import("torch");
    if (Config::use_shared_tensor_infer) {
      torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
      torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()
          ->init(sta::DeviceManager::GetNumVisibleGpu());
      torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()
          ->SetCurrentAllocator();
      LOG(INFO) << "[LLM Server] torch cuda allocator initialized";
    }
    py_module_ = bp::import("llm");
  
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
    
    PyThreadState *pts = PyThreadState_Get();
    PyEval_ReleaseThread(pts);
  } catch (const bp::error_already_set&) {
    PyErr_Print();
    LOG(FATAL) << "[LLMWrapper] create infer backend failed";
  }
  thread_.reset(new std::thread(&LLMWrapper::Inference, this, barrier));
}

void LLMWrapper::Inference(pthread_barrier_t* barrier) {
  pthread_barrier_wait(barrier);
  // llm_infer_.attr("serving_loop");
  try {
    PyGILState_Ensure();
    if (!PyObject_HasAttrString(llm_infer_.ptr(), "serving_loop")) {
      LOG(FATAL) << "[LLMWrapper] rank " << rank_
                << " serving_loop not found";
    }
    // PyGILState_Release();
    auto serving_loop = llm_infer_.attr("serving_loop");
    serving_loop();
  } catch (const bp::error_already_set &) {
    PyErr_Print();
    LOG(FATAL) << "[LLMWrapper] rank " << rank_ 
               << " serving_loop failed";
  }
  
  LOG(INFO) << "[LLMWrapper] rank " << rank_ << " exit";
}

} // namespace colserve

