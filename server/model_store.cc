#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "model_store.h"

namespace colserve {

std::unique_ptr<ModelStore> ModelStore::model_store_;

ModelStore* ModelStore::Get() {
  if (model_store_ == nullptr) {
    LOG(FATAL) << "ModelStore not initialized";
  }
  return model_store_.get();
}

void ModelStore::Init(const std::filesystem::path &model_store_path) {
  DLOG(INFO) << "ModelStore start initializing";
  model_store_ = std::make_unique<ModelStore>();
  model_store_->models_["dummy"] = std::make_unique<Model>();

  std::map<std::string, std::string> models;
  std::ifstream config_file(model_store_path / "config");
  if (config_file.good()) {
    for (std::string line; std::getline(config_file, line);) {
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string model_name, device;
      iss >> model_name >> device;
      models[model_name] = device;
    }
  }

  // mod.so, mod.json, mod.params should be in the model directory
  for (const auto &entry : std::filesystem::directory_iterator(model_store_path)) {
    if (entry.is_directory() && (models.empty() || models.count(entry.path().filename().string()))) {
      auto model_name = entry.path().filename().string();
      auto model_path = entry.path();
      DLDevice device;
      if (models[model_name] == "cpu") {
        device = DLDevice{kDLCPU, 0};
      } else if (models[model_name] == "cuda") {
        device = DLDevice{kDLCUDA, 0};
      } else {
        LOG(FATAL) << model_name << " unsupport device type " << models[model_name];
      }
      model_store_->models_[model_name] = 
          std::make_unique<Model>(model_name, model_path, device);
      LOG(INFO) << "ModelStore: "<< "Add " << model_name << ":" << models[model_name] << " into ModelStore";
    }
  }

  LOG(INFO) << "ModelStore initialized";
}

Model* ModelStore::GetModel(const std::string &name) {
  auto it = models_.find(name);
  if (it == models_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

Model::Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device) 
    : name_(name), device_(device) {
  // DLOG(INFO) << "Model " << name << " start initializing from " << model_path;
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
  }

  // TODO get output info
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
  }
}

bool Model::Inference() {
  LOG(INFO) << "Model " << name_ << " inference thread start";
  while (true) {
    // TODO dynamic batching
    auto jobs = job_queue_.GetBatch(1, 100);
    // tvm::runtime::NDArray input;

    bool err = false;
    for (auto& input: input_info_) {
      auto& input_id = input.first;
      err = SetInput(input_id, jobs);
    }

    run_();

    for (auto& output : output_info_) {
      auto& output_id = output.first;
      err = GetOutput(output_id, jobs);
    }

    for (size_t i = 0; i < jobs.size(); i++) {
      auto data = jobs[i]->GetInferData();
      data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
      LOG(INFO) << jobs[i] << " finished";
    }
  }
}

bool Model::SetInput(const std::string &input_id, const std::vector<std::shared_ptr<Job>> &jobs) {
  // check input shape
  auto input_dtype = jobs[0]->GetInferData()->GetInputDType();
  auto batch_shape = jobs[0]->GetInferData()->GetInputShape();
  batch_shape[0] = jobs.size();
  CHECK_EQ(batch_shape.size(), input_info_[input_id].first.size());
  for (size_t i = 0; i < batch_shape.size(); i++) {
    CHECK_EQ(batch_shape[i], input_info_[input_id].first[i])
        << "input shape mismatch";
  }

  tvm::runtime::NDArray input_arr = get_input_(input_id, device_);
  CHECK_EQ(input_arr.DataType(), tvm::runtime::DataType(input_dtype)) << "input dtype mismatch";

  auto copy_batch_input_fn = [&jobs](void* data) {
    size_t offset = 0;
    for (size_t i = 0; i < jobs.size(); i++) {
      std::memcpy(static_cast<char*>(data) + offset,
                  jobs[i]->GetInferData()->GetInput(),
                  jobs[i]->GetInferData()->GetInputBytes());
    }
  };

  if (device_.device_type == kDLCPU) {
      copy_batch_input_fn(input_arr->data);
  } else if (device_.device_type == kDLCUDA) {
    auto input_cpu = tvm::runtime::NDArray::Empty(
        input_info_[input_id].first, input_dtype, DLDevice{kDLCUDAHost, 0});
    copy_batch_input_fn(input_cpu->data);
    input_arr.CopyFrom(input_cpu);
  } else {
    LOG(FATAL) << "unsupport device type " << device_.device_type;
  }
  return true;
}

bool Model::GetOutput(const std::string &output_id, const std::vector<std::shared_ptr<Job>> &jobs) {
  tvm::runtime::NDArray output_arr = get_output_(output_id);
  tvm::runtime::NDArray output_cpu;
  if (device_.device_type == kDLCPU) {
    output_cpu = output_arr;
  } else if (device_.device_type == kDLCUDA) {
    output_cpu = output_arr.CopyTo(DLDevice{kDLCUDAHost, 0});
  } else {
    LOG(FATAL) << "unsupport device type " << device_.device_type;
  }
  CHECK_EQ(output_cpu.Shape()[0], jobs.size()) << "batch size mismatch";
  for (size_t i = 0; i < jobs.size(); i++) {
    auto data = jobs[i]->GetInferData();
    

    // // TODO hard code for mnist here, should be generalized
    // int number = -1;
    // float max_prob = -std::numeric_limits<float>::max();
    // for (int i = 0; i < 10; i++) {
    //   if (static_cast<float*>(output_cpu->data)[i] > max_prob) {
    //     number = i;
    //     max_prob = static_cast<float*>(output_cpu->data)[i];
    //   }
    // }
    // LOG(INFO) << jobs[i] << " number " << number <<  " prob " << max_prob;
    // data->GetResponse().set_result(
    //     "result " + std::to_string(number));
  }
  return true;
}


} // namespace colserve