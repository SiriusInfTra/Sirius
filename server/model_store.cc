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

  for (const auto &entry : std::filesystem::directory_iterator(model_store_path)) {
    if (entry.is_directory()) {
      auto model_name = entry.path().filename().string();
      auto model_path = entry.path();
      auto device = DLDevice{kDLCPU, 0};
      model_store_->models_[model_name] = 
          std::make_unique<Model>(model_name, model_path, device);
      LOG(INFO) << "Add " << model_name << " into ModelStore";
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
  DLOG(INFO) << "Model " << name << " start initializing from " << model_path;
  rmod_ = tvm::runtime::Module::LoadFromFile((model_path / "mod.so").c_str(), "so");

  std::ifstream json((model_path / "mod.json").c_str());
  std::string json_str((std::istreambuf_iterator<char>(json)),
                       std::istreambuf_iterator<char>());
  json.close();

  auto f = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  graph_executor_ = (*f)(json_str, rmod_, static_cast<int>(device.device_type), device.device_id);
  
  std::ifstream params((model_path / "mod.params").c_str(), std::ios::binary);
  const std::string params_str((std::istreambuf_iterator<char>(params)),
                               std::istreambuf_iterator<char>());
  params.close();

  graph_executor_.GetFunction("load_params")(params_str);
  LOG(INFO) << "Model " << name << " initialized";


  set_input_ = graph_executor_.GetFunction("set_input");
  run_ = graph_executor_.GetFunction("run");
  get_output_ = graph_executor_.GetFunction("get_output");

  InitMetaInfo();

  thread_.reset(new std::thread{&Model::inference, this});
}

bool Model::AddJob(network::InferHandler::InferData* data) {
  // data->GetResponse().set_result("add into queue");
  // data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
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
    std::vector<int> shape{shape_tuple.begin(), shape_tuple.end()};
    auto dtype = tvm::runtime::GetRef<tvm::runtime::String>(
        dtype_info[kv.first].as<tvm::runtime::StringObj>());
    input_info_[kv.first] = std::make_pair(shape, dtype);
    std::stringstream ss;
    for (auto &dim : shape) ss << dim << " ";
    DLOG(INFO) << "model " << name_ << " " << kv.first << " shape " << ss.str()  << " dtype " << dtype;
  }

  
}

bool Model::inference() {
  LOG(INFO) << "Model " << name_ << " inference thread start";
  while (true) {
    // TODO dynamic batching
    auto jobs = job_queue_.GetBatch(1, 100);
    // tvm::runtime::NDArray input;
    auto batch_shape = jobs[0]->GetInferData()->GetInputShape();
    batch_shape[0] = jobs.size();
    auto input_cpu = tvm::runtime::NDArray::Empty(
        batch_shape, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});
    size_t offset = 0;
    for (size_t i = 0; i < jobs.size(); i++) {
      std::memcpy(static_cast<char*>(input_cpu->data) + offset,
                  jobs[i]->GetInferData()->GetInput(),
                  jobs[i]->GetInferData()->GetInputBytes());
    }
    auto intput_dev = input_cpu.CopyTo(device_);
    set_input_("input", intput_dev);
    run_();

    auto output_dev = tvm::runtime::NDArray::Empty(
        {batch_shape[0], 1000}, DLDataType{kDLFloat, 32, 1}, device_);
    get_output_(0, output_dev);
    auto output_cpu = output_dev.CopyTo(DLDevice{kDLCPU, 0});
    for (size_t i = 0; i < jobs.size(); i++) {
      auto data = jobs[i]->GetInferData();
      data->GetResponse().set_result(
          "result " + std::to_string(static_cast<float*>(output_cpu->data)[0]));
      data->GetResponder().Finish(data->GetResponse(), grpc::Status::OK, data);
      LOG(INFO) << "inference " << data->GetModelName() << " " << data->GetId() << " finished";
    }
  }
}


} // namespace colserve