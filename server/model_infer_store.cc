#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

#include "model_infer_store.h"
#include "model_train_store.h"
#include "controller.h"
#include "config.h"

namespace colserve {

std::unique_ptr<ModelInferStore> ModelInferStore::model_infer_store_;

ModelInferStore* ModelInferStore::Get() {
  if (model_infer_store_ == nullptr) {
    LOG(FATAL) << "ModelInferStore not initialized";
  }
  return model_infer_store_.get();
}

void ModelInferStore::Init(const std::filesystem::path &infer_store_path) {
  DLOG(INFO) << "ModelInferStore start initializing";
  model_infer_store_ = std::make_unique<ModelInferStore>();
  model_infer_store_->models_["dummy"] = std::make_unique<Model>();

  // model_name -> [key -> value]
  std::map<std::string, std::map<std::string, std::string>> models;
  std::ifstream config_file(infer_store_path / "config");
  if (config_file.good()) {
    std::string cur_model;
    for (std::string line; std::getline(config_file, line);) {
      if (line.empty()) continue;
      if (line[0] == '#') continue;
      std::istringstream iss(line);
      if (line[0] != ' ') {
        iss >> cur_model;
        models[cur_model] = {};
      } else {
        std::string key, value;
        iss >> key >> value;
        models[cur_model][key] = value;
      }
    }
    for (auto &model: models) {
      std::stringstream ss;
      ss << "[" << model.first << "] "; 
      CHECK(model.second.count("path"));
      CHECK(model.second.count("device"));
      CHECK(model.second.count("batch-size"));
      for (auto &kv : model.second) {
        ss << kv.first << "=" << kv.second << " ";
      }
      LOG(INFO) << "[ModelInferStore] Read from config file: " << ss.str();
    }
  }

  // mod.so, mod.json, mod.params should be in the model directory
  for (auto &model : models) {
    auto model_name = model.first;
    auto model_path = infer_store_path / model.second["path"];
    auto model_device = model.second["device"];
    size_t batch_size = std::stoi(model.second["batch-size"]);
    CHECK(std::filesystem::exists(model_path)) << "ModelInferStore: " << model_path << " not exist";
    CHECK(std::filesystem::is_directory(model_path)) << model_path << " is not a directory";

    DLDevice device;
    if (model_device == "cpu") {
      device = DLDevice{kDLCPU, 0};
    } else if (model_device == "cuda") {
      device = DLDevice{kDLCUDA, 0};
    } else {
      LOG(FATAL) << model_name << " unsupport device type " << model_device;
    }
    model_infer_store_->models_[model_name] = 
        std::make_unique<Model>(model_name, model_path, device, batch_size);
    LOG(INFO) << "ModelInferStore: "<< "Add " << model_name << ":" << model_device
              << ", batch-size=" << batch_size;
  }

  LOG(INFO) << "ModelInferStore initialized";
}

Model* ModelInferStore::GetModel(const std::string &name) {
  auto it = models_.find(name);
  if (it == models_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

size_t ModelInferStore::NumJobs() {
  size_t num_jobs = 0;
  for (auto &model : models_) {
    num_jobs += model.second->NumJobs();
  }
  return num_jobs;
}

Model::Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device, size_t batch_size)
    : name_(name), device_(device), batch_size_(batch_size) {
  // DLOG(INFO) << "Model " << name << " start initializing from " << model_path;
  CHECK(std::filesystem::exists(model_path / "mod.so"));
  CHECK(std::filesystem::exists(model_path / "mod.json"));
  CHECK(std::filesystem::exists(model_path / "mod.params"));

  rmod_ = tvm::runtime::Module::LoadFromFile((model_path / "mod.so").c_str(), "so");

  std::ifstream json{(model_path / "mod.json").c_str()};
  std::string json_str{(std::istreambuf_iterator<char>(json)),
                       std::istreambuf_iterator<char>()};
  json.close();

  auto f = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  graph_executor_ = (*f)(json_str, rmod_, static_cast<int>(device.device_type), device.device_id);
  
  std::ifstream params{(model_path / "mod.params").c_str(), std::ios::binary};
  CHECK(!params.fail()) << "Fail to open " << (model_path / "mod.params").string();
  const std::string params_str{(std::istreambuf_iterator<char>(params)),
                               std::istreambuf_iterator<char>()};
  params.close();

  TVMByteArray params_arr;
  params_arr.data = params_str.c_str();
  params_arr.size = params_str.length();
  graph_executor_.GetFunction("load_params")(params_arr);
  LOG(INFO) << "Model: " << name << " initialized";

  InitMetaInfo();

  if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    reset_storage_ = graph_executor_.GetFunction("reset_storage");
    reset_storage_();

    alloc_storage_ = graph_executor_.GetFunction("alloc_storage");
    // alloc_storage_();

    // auto t0 = std::chrono::steady_clock::now();
    pipeline_load_params_ = graph_executor_.GetFunction("pipeline_load_params");
    // pipeline_load_params_();
    // auto t1 = std::chrono::steady_clock::now();
    // LOG(INFO) << "Model: " << name << " pipeline_load_params " 
    //           << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
    pipeline_run_ = graph_executor_.GetFunction("pipeline_run");

    LOG(INFO) << "[Model]: " << name << " task switch by pipelining load/run";
  }


  // get tvm packed functions for inference
  set_input_ = graph_executor_.GetFunction("set_input");
  get_input_ = graph_executor_.GetFunction("get_input");
  run_ = graph_executor_.GetFunction("run");
  get_output_ = graph_executor_.GetFunction("get_output");

  thread_.reset(new std::thread{&Model::Inference, this});
}

bool Model::AddJob(network::InferHandler::InferData* data) {
  if (name_ == "dummy") {
    data->GetResponse().set_result("dummy result");
    data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    return true;
  }
  Controller::Get()->InferRequestInc();
  // InterruptTrain check whether to interrupt train
  Controller::Get()->InterruptTrain(); 
  return job_queue_.Put(std::make_shared<InferJob>(data));
}

void Model::InitMetaInfo() {
  using TVMMap = tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef>;
  TVMMap tvm_input_info =  graph_executor_.GetFunction("get_input_info")();
  auto shape_info = tvm::runtime::GetRef<TVMMap>(tvm_input_info["shape"].as<tvm::runtime::MapNode>());
  auto dtype_info = tvm::runtime::GetRef<TVMMap>(tvm_input_info["dtype"].as<tvm::runtime::MapNode>());
  for (const auto &kv : shape_info) {
    auto shape_tuple = tvm::runtime::GetRef<tvm::runtime::ShapeTuple>(
        kv.second.as<tvm::runtime::ShapeTupleObj>());
    auto dtype = tvm::runtime::GetRef<tvm::runtime::String>(
        dtype_info[kv.first].as<tvm::runtime::StringObj>());
    input_info_[kv.first] = std::make_pair(shape_tuple, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape_tuple.size(); i++) 
      ss << shape_tuple[i] << " ";
    DLOG(INFO) << "Model Input: " << name_ << " " << kv.first << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape_tuple[0], batch_size_) << "batch size mismatch";
  }

  TVMMap tvm_output_info =  graph_executor_.GetFunction("get_output_info")();
  shape_info = tvm::runtime::GetRef<TVMMap>(tvm_output_info["shape"].as<tvm::runtime::MapNode>());
  dtype_info = tvm::runtime::GetRef<TVMMap>(tvm_output_info["dtype"].as<tvm::runtime::MapNode>());
  for (const auto &kv : shape_info) {
    auto shape_tuple = tvm::runtime::GetRef<tvm::runtime::ShapeTuple>(
        kv.second.as<tvm::runtime::ShapeTupleObj>());
    auto dtype = tvm::runtime::GetRef<tvm::runtime::String>(
        dtype_info[kv.first].as<tvm::runtime::StringObj>());
    output_info_[kv.first] = std::make_pair(shape_tuple, dtype);
    std::stringstream ss;
    for (size_t i = 0; i < shape_tuple.size(); i++)
      ss << shape_tuple[i] << " ";
    DLOG(INFO) << "Model Output: " << name_ << " " << kv.first << " shape [ " << ss.str()  << "] dtype " << dtype;
    CHECK_EQ(shape_tuple[0], batch_size_) << "batch size mismatch";
  }
}

bool Model::Inference() {
  LOG(INFO) << "Model " << name_ << " inference thread start";
  while (true) {  
    // TODO dynamic batching
    auto jobs = job_queue_.GetBatch(batch_size_, 10);
    LOG(INFO) << "[Model Inference] GetBatch " << jobs.size() << "/" << batch_size_;

    auto infer_begin = std::chrono::steady_clock::now();

    double wait_train_stop_ms = -1;
    if (Config::serve_mode == ServeMode::kTaskSwitchL1 
        || Config::serve_mode == ServeMode::kTaskSwitchL2
        || Config::serve_mode == ServeMode::kTaskSwitchL3) {
      auto begin = std::chrono::steady_clock::now();
      Controller::Get()->WaitTrainNotRunning();
      wait_train_stop_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
      // DLOG(INFO) << "ModelInferStore: wait for train stop "
      //            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }

    std::future<tvm::runtime::TVMRetValue> future;
    if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
      alloc_storage_();
      // pipeline_load_params_();
      // std::this_thread::sleep_for(std::chrono::seconds(1));
      auto t0 = std::chrono::steady_clock::now();
      future = std::async(std::launch::async, pipeline_load_params_);
      auto t1 = std::chrono::steady_clock::now();
      LOG(INFO) << "pipeline load param async " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms";
    }

    // tvm::runtime::NDArray input;

    bool err = false;
    size_t idx = 0;

    double set_input_ms;
    {
      auto begin = std::chrono::steady_clock::now();
      for (auto& input: input_info_) {
        auto& input_id = input.first;
        err = SetInput(idx++, input_id, jobs);
      }
      auto end = std::chrono::steady_clock::now();
      set_input_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      // DLOG(INFO) << "Inference SetInput: " << std::chrono::duration<double, std::milli>(end - begin).count();
    }

    double infer_ms;
    {
      auto begin = std::chrono::steady_clock::now();
      if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
        pipeline_run_();
        // run_();
      } else {
        run_();
      }
      auto end = std::chrono::steady_clock::now();
      infer_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      // DLOG(INFO) << "Inference run: " << std::chrono::duration<double, std::milli>(end - begin).count();
    }

    double get_output_ms;
    idx = 0;
    {
      auto begin = std::chrono::steady_clock::now();
      for (auto& output : output_info_) {
        for (auto& job : jobs)
          job->GetInferData()->AddOuput();
        auto& output_id = output.first;
        err = GetOutput(idx++, output_id, jobs);
      }
      auto end = std::chrono::steady_clock::now();
      get_output_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      // DLOG(INFO) << "Inference GetOutput: " << std::chrono::duration<double, std::milli>(end - begin).count();
    }
    auto infer_end = std::chrono::steady_clock::now();

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "[Inference]: " << name_ << " "
       << "set_input_ms=" << set_input_ms << " "
       << "infer_ms=" << infer_ms << " "
       << "get_output_ms=" << get_output_ms;
    if (wait_train_stop_ms != -1) {
      ss << " wait_train_stop_ms=" << wait_train_stop_ms;
    }
    ss << " total_infer_ms=" << std::chrono::duration<double, std::milli>(infer_end - infer_begin).count();
    DLOG(INFO) << ss.str();

    for (auto& job : jobs) {
      auto data = job->GetInferData();
      LOG(INFO) << job << " finished";
      data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
    }

    Controller::Get()->InferResponseInc(jobs.size());
    if (Config::serve_mode == ServeMode::kTaskSwitchL1 && Controller::Get()->IsInferIdle()) {
      Controller::Get()->ResumeTrain();
    } else if (Config::serve_mode == ServeMode::kTaskSwitchL3 && job_queue_.NumJobs() == 0) {
      reset_storage_();
    }
  }
}

bool Model::SetInput(size_t idx, const std::string &input_id, const std::vector<std::shared_ptr<Job>> &jobs) {
  // check input shape
  auto input_dtype = jobs[0]->GetInferData()->GetInputDType(idx);
  auto batch_shape = jobs[0]->GetInferData()->GetInputShape(idx);
  batch_shape[0] = jobs.size();

  CHECK_EQ(batch_shape.size(), input_info_[input_id].first.size()) << "input shape dimension mismatch";
  CHECK_LE(batch_shape[0], input_info_[input_id].first[0]) << "out of model batch size";
  for (size_t i = 1; i < batch_shape.size(); i++) {
    CHECK_EQ(batch_shape[i], input_info_[input_id].first[i])
        << "input shape mismatch";
  }

  tvm::runtime::NDArray input_arr = get_input_(input_id, device_);
  CHECK_EQ(input_arr.DataType(), tvm::runtime::DataType(input_dtype)) << "input dtype mismatch";

  auto copy_batch_input_fn = [&jobs](size_t idx, void* data) {
    size_t offset = 0;
    for (auto job : jobs) {
      auto bytes= job->GetInferData()->GetInputBytes(idx);
      std::memcpy(static_cast<char*>(data) + offset,
                  job->GetInferData()->GetInputData(idx),
                  bytes);
      offset += bytes;
    }
  };

  if (device_.device_type == kDLCPU) {
      copy_batch_input_fn(idx, input_arr->data);
  } else if (device_.device_type == kDLCUDA) {
    auto input_cpu = tvm::runtime::NDArray::Empty(
        input_info_[input_id].first, input_dtype, DLDevice{kDLCUDAHost, 0});
    copy_batch_input_fn(idx, input_cpu->data);
    input_arr.CopyFrom(input_cpu);
  } else {
    LOG(FATAL) << "unsupport device type " << device_.device_type;
  }
  return true;
}

bool Model::GetOutput(size_t idx, const std::string &output_id, const std::vector<std::shared_ptr<Job>> &jobs) {
  tvm::runtime::NDArray output_cpu; // = get_output_(output_id);
  if (device_.device_type == kDLCPU) {
    output_cpu = get_output_(output_id);
  } else {
    tvm::runtime::NDArray output_dev = get_output_(output_id);
    output_cpu = output_dev.CopyTo(DLDevice{kDLCPU, 0});
  }

  CHECK_LE(jobs.size(), output_cpu.Shape()[0]) << "out of model batch size";
  std::vector<int64_t> shape{output_cpu.Shape().begin(), output_cpu.Shape().end()};
  shape[0] = 1;

  size_t offset = 0;
  for (auto& job : jobs) {
    auto data = job->GetInferData();
    data->SetOutputShape(idx, shape);
    data->SetOutputDType(idx, tvm::runtime::DLDataType2String(output_cpu.DataType()));
    // copy output to infer data
    auto bytes = tvm::runtime::GetDataSize(*output_cpu.operator->()) / output_cpu.Shape()[0];
    auto output = data->MutableOutputData(idx);
    output->resize(bytes);
    // output_cpu.CopyToBytes(output->data(), bytes);
    std::memcpy(output->data(), static_cast<char*>(output_cpu->data) + offset, bytes);
    offset += bytes;
  }
  return true;
}


} // namespace colserve