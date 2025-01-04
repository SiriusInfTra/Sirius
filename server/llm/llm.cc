#include <server/logging_as_glog.h>
#include <server/config.h>
#include <server/llm/llm.h>
#include <server/llm/llm_util.h>
// #include <server/llm/llm_torch_allocator_plugin.h>
#include <server/llm/kv_cache_pool.h>
#include <boost/json.hpp>

#include <boost/format.hpp>
#include <boost/range/irange.hpp>
#include <filesystem>

#include <memory>
#include <string>

namespace colserve {

std::unique_ptr<LLMServer> LLMServer::llm_server_ = nullptr;

BOOST_PYTHON_MODULE(llm_server)
{
  bp::def("is_running", &LLMServer::IsRunning);
  bp::def("get_llm_requests", &LLMServer::GetLLMRequests);
  bp::def("finish_llm_request", &LLMServer::FinishLLMRequest);
  bp::def("check_kv_cache_block_nbytes", &KVCachePool::CheckKVCacheBlockNbytes);
  bp::def("get_num_gpu_kv_cache_blocks", &KVCachePool::GetNumGpuKVCacheBlocks);
  bp::def("init_kv_cache", &KVCachePool::InitKVCache);
  bp::def("info", &CallGLOG_INFO);
  bp::def("dinfo", &CallGLOG_DINFO);

  bp::class_<LLMRequest>("LLMRequest")
      .def_readwrite("request_id", &LLMRequest::request_id)
      .def_readwrite("prompt", &LLMRequest::prompt)
      .def_readwrite("max_tokens", &LLMRequest::max_tokens)
      .def("__repr__", +[](const LLMRequest& self) -> std::string {
          return str(boost::format(
              "LLMRequest(request_id=%d, prompt=\"%s\", max_tokens=%d)") 
              % self.request_id 
              % self.prompt 
              % self.max_tokens);
      }); 
  bp::class_<LLMRequestMetric>("LLMRequestMetric")
      .def_readwrite("num_prompt_token", &LLMRequestMetric::num_prompt_token)
      .def_readwrite("num_output_token", &LLMRequestMetric::num_output_token)
      .def_readwrite("queue_ms", &LLMRequestMetric::queue_ms)
      .def_readwrite("prefill_ms", &LLMRequestMetric::prefill_ms)
      .def_readwrite("decode_ms", &LLMRequestMetric::decode_ms)
      .def(bp::init<int,int,double,double,double>(
          (bp::arg("num_prompt_token"),
           bp::arg("num_output_token"),
           bp::arg("queue_ms"),
           bp::arg("prefill_ms"),
           bp::arg("decode_ms"))))
      .def("__repr__", +[](const LLMRequestMetric& self) -> std::string {
          return str(boost::format(
              "LLMRequestMetric(num_prompt_token=%d, num_output_token=%d, "
              "queue_ms=%.2f, prefill_ms=%.2f, decode_ms=%.2f)") 
              % self.num_prompt_token 
              % self.num_output_token 
              % self.queue_ms 
              % self.prefill_ms 
              % self.decode_ms);
      });

  bp::to_python_converter<std::vector<LLMRequest>, 
                          LLMRequestsConvert, false>();
}

void LLMServer::Init() {
  CHECK(IsLLMModel(Config::llm_model_name));
  KVCachePool::Init();
  try {
    llm_server_ = std::make_unique<LLMServer>();
  } catch (boost::python::error_already_set const &) {
    PyErr_Print();
    LOG(ERROR) << "Error importing python module";
  }
  LOG(INFO) << "[LLMServer] initialized";
}

void LLMServer::Shutdown() {
  CHECK(llm_server_ != nullptr);
  llm_server_->running_.store(false, std::memory_order_release);
  for (auto &wrapper : llm_server_->llm_wrappers_) {
    wrapper->Shutdown();
  }
  LOG(INFO) << "[LLMServer] shutdown";
}

bool LLMServer::IsLLMModel(const std::string &model_name) {
  if (model_name.find("llama") != std::string::npos) {
    return true;
  } else if (model_name.find("opt") != std::string::npos) {
    return true;
  }
  return false;
}

bool LLMServer::AddJob(network::InferHandler::InferData *data) {
  if (data->GetModelName() != Config::llm_model_name) {
    LOG(INFO) << "[LLMServer] model " << data->GetModelName() 
              << " mismatch with " << Config::llm_model_name;
    return false;
  }

  CHECK(llm_server_ != nullptr);
  CHECK(llm_server_->llm_wrappers_.size() > 0);
  // llm_server_->llm_wrappers_.front()->AddJob(data);
  llm_server_->job_queue_.Put(std::make_shared<LLMInferJob>(data));
  return true;
}

std::vector<LLMRequest>
LLMServer::GetLLMRequests(int batch_size, int timeout_ms, bool block) {
  CHECK(llm_server_ != nullptr);
  while (true) {
    auto jobs = llm_server_->job_queue_.GetBatch(batch_size, 10, timeout_ms);
    if (jobs.size() > 0) {
      std::vector<LLMRequest> llm_reqs;
      for (auto & job : jobs) {
        auto llm_job = std::dynamic_pointer_cast<LLMInferJob>(job);
        auto req_id = job->GetInferData()->GetId();        
        // llm_server_->flight_reqs_[req_id] = llm_job;
        llm_server_->AddFlightRequest(req_id, llm_job);
        llm_reqs.emplace_back(LLMRequest{
            .request_id = req_id,
            .prompt = std::string{llm_job->GetPrompt()},
            .max_tokens = llm_job->GetMaxTokens()
          });
      }
      return llm_reqs;
    } else if (!block) {
      return {};
    }
    if (!llm_server_->IsRunning()) {
      return {};
    }
  }
}

void LLMServer::AddFlightRequest(uint64_t request_id, 
                                 std::shared_ptr<LLMInferJob> job) {
  std::unique_lock lock{mut_};
  CHECK(flight_reqs_.find(request_id) == flight_reqs_.end())
      << "[LLMServer] request " << request_id << " already exists";
  flight_reqs_[request_id] = job;
}

void LLMServer::FinishLLMRequest(
    int request_id, std::string output, 
    LLMRequestMetric metric) {
  CHECK(llm_server_ != nullptr);
  std::unique_lock lock{llm_server_->mut_};

  auto it = llm_server_->flight_reqs_.find(request_id);
  if (it == llm_server_->flight_reqs_.end()) {
    LOG(ERROR) << "[LLMServer] request " << request_id << " not found";
    return;
  }

  LOG_IF_EVERY_N(INFO, 
      Config::llm_show_gen_result, Config::llm_show_gen_result_period) 
    << "[LLMServer] Finish request " << request_id 
    << ", num_prompt_token " << metric.num_prompt_token
    << ", num_output_token " << metric.num_output_token
    << ", generate output: " << output;

  auto job = it->second;
  job->RecordFinished();
  job->RecordProfile(metric);

  auto data = job->GetInferData();
  // generate result
  data->AddOuput();
  data->SetOutputShape(0, {static_cast<int64_t>(output.size())});
  data->SetOutputDType(0, "char");
  auto output_data = data->MutableOutputData(0);
  output_data->resize(output.size());
  std::memcpy(output_data->data(), output.data(), output.size());

  // llm infer info
  boost::json::object llm_info;
  llm_info["num_prompt_token"] = metric.num_prompt_token;
  llm_info["num_output_token"] = metric.num_output_token;
  llm_info["prefill_ms"] = metric.prefill_ms;
  llm_info["decode_ms"] = metric.decode_ms;
  auto info_str = boost::json::serialize(llm_info);
  data->AddOuput();
  data->SetOutputShape(1, {static_cast<int64_t>(info_str.size())});
  data->SetOutputDType(1, "char");
  auto info_data = data->MutableOutputData(1);
  info_data->resize(info_str.size());
  std::memcpy(info_data->data(), info_str.data(), info_str.size());

  job->RecordFinished();
  data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
  llm_server_->flight_reqs_.erase(it);
}

LLMServer::LLMServer() : running_(true) {
  PyInit();
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, nullptr, 1 + llm_wrappers_.size());
  for (auto i : boost::irange(sta::DeviceManager::GetNumVisibleGpu())) {
    llm_wrappers_.push_back(std::make_unique<LLMWrapper>(
        i, py_module_, &barrier));
  }

  // release GIL
  main_py_ts_ = PyEval_SaveThread();
  pthread_barrier_wait(&barrier);
}

void LLMServer::PyInit() {
  try {
    PyImport_AppendInittab("llm_server", &PyInit_llm_server);
    Py_Initialize();
    LOG(INFO) << "[LLMServer] python initialized";

    // main_py_ts_ = PyThreadState_GET();
    // add the current directory to the python path
    bp::exec((boost::format(
      "import sys\nsys.path.append(%s)") % 
      (std::filesystem::path(__FILE__).parent_path() / "python")
    ).str().c_str());
    bp::exec((boost::format(
      "import sys\nprint(sys.path)\n")
    ).str().c_str());

    // bp::import("torch");
    // if (Config::use_shared_tensor_infer) {
    //   torch::cuda::CUDAColAllocator::CUDAColAllocator::Init();
    //   torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()
    //       ->init(sta::DeviceManager::GetNumVisibleGpu());
    //   torch::cuda::CUDAColAllocator::CUDAColAllocator::Get()
    //       ->SetCurrentAllocator();
    //   LOG(INFO) << "[LLM Server] torch cuda allocator initialized";
    // }
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
        Config::llm_max_model_len,
        Config::llm_max_seq_len
    );
  } catch (const bp::error_already_set&) {
    PyErr_Print();
    LOG(FATAL) << "[LLMWrapper] create infer backend failed";
  }
  thread_.reset(new std::thread(&LLMWrapper::Inference, this, barrier));
}

void LLMWrapper::Inference(pthread_barrier_t* barrier) {
  pthread_barrier_wait(barrier);
  // llm_infer_.attr("serving_loop");
  LOG(INFO) << "[LLMWrapper] rank " << rank_ << " inference start";
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

void LLMWrapper::Shutdown() {
  if (thread_ != nullptr) {
    thread_->join();
  }
}

} // namespace colserve

