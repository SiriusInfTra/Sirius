#include "logging_as_glog.h"
#include <thread>
#include <pthread.h> 
#include "../model_infer_store.h"
#include "grcp_server.h"
#include "../model_train_store.h"
#include "../controller.h"
#include "../config.h"


namespace colserve {
namespace network {

void GRPCServer::Start(const std::string &addr) {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

  builder.RegisterService(&service_);
  common_cq_ = builder.AddCompletionQueue();
  infer_cq_ = builder.AddCompletionQueue();
  train_cq_ = builder.AddCompletionQueue();

  server_ = builder.BuildAndStart();

  common_handler_ = std::make_unique<CommonHandler>(&service_, common_cq_.get());
  infer_handler_ = std::make_unique<InferHandler>(&service_, infer_cq_.get());
  train_handler_ = std::make_unique<TrainHandler>(&service_, train_cq_.get());

  common_handler_->Start();
  infer_handler_->Start();
  train_handler_->Start();

  LOG(INFO) << "GRPCServer start at " << addr;
}

void GRPCServer::Stop() {
  common_handler_->Stop();
  infer_handler_->Stop();
  train_handler_->Stop();

  common_cq_->Shutdown();
  infer_cq_->Shutdown();
  train_cq_->Shutdown();
  server_->Shutdown(); 
}

GRPCServer::~GRPCServer() {
  Stop();
}


void CommonHandler::Start() {
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, 2);
  thread_.reset(new std::thread([this, &barrier]() {
    pthread_barrier_wait(&barrier);
    SetupCallData();

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      auto call_data = static_cast<GRPCServer::CallData*>(tag);
      CHECK(ok) << "CommonHandler get CallData failed";
      if (!call_data->Process(ok)) {
        delete call_data;
      }
    }
  }));
  pthread_barrier_wait(&barrier);
  LOG(INFO) << "CommonHandler start";
}

void CommonHandler::Stop() {
  thread_->join();
  LOG(INFO) << "CommonHandler stop";
}

void CommonHandler::SetupCallData() {
  auto register_get_server_status = [this](
      CommonData<EmptyRequest, ServerStatus>* data) {
    this->service_->RequestGetServerStatus(
        &data->ctx_, &data->request_, &data->responder_, 
        data->cq_, data->cq_, (void*)data);
  };
  auto exec_get_server_status = [this](
      CommonData<EmptyRequest, ServerStatus>* data) {
    data->response_.set_status("healthy");
    data->responder_.Finish(data->response_, grpc::Status::OK, (void*)data);
  };
  new CommonData<EmptyRequest, ServerStatus>{0, "GetServerStatus", service_, cq_,
                                             register_get_server_status,
                                             exec_get_server_status};
}

void InferHandler::Start() {
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, 2);
  thread_.reset(new std::thread([this, &barrier]() {
    pthread_barrier_wait(&barrier);
    InferData* infer_data = new InferData{0, "InferRequest", service_, cq_};

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      infer_data = static_cast<InferData*>(tag);
      CHECK(ok) << "InferHandler get CallData failed";
      if (!infer_data->Process(ok)) {
        delete infer_data;
      }
    }
  }));
  pthread_barrier_wait(&barrier);
  LOG(INFO) << "InferHandler start";
}

void InferHandler::Stop() {
  thread_->join();
  LOG(INFO) << "InferHandler stop";
}



DLDataType InferHandler::InferData::GetInputDType(size_t i) {
  if (request_.inputs(i).dtype() == "float32") {
    return DLDataType{kDLFloat, 32, 1};
  } else if (request_.inputs(i).dtype() == "int8") {
    return DLDataType{kDLInt, 8, 1};
  } else {
    LOG(FATAL) << "InferData: " << "Unknown input dtype " << request_.inputs(i).dtype();
  }
}

std::vector<int64_t> InferHandler::InferData::GetInputShape(size_t i) {
  std::vector<int64_t> shape;
  for (size_t j = 0; j < request_.inputs(i).shape().size(); ++j) {
    shape.push_back(request_.inputs(i).shape(j));
  }
  return shape;
}

const char* InferHandler::InferData::GetInputData(size_t i) {
  return request_.inputs(i).data().c_str();
}

size_t InferHandler::InferData::GetInputBytes(size_t i) {
  return request_.inputs(i).data().size();
}

void InferHandler::InferData::AddOuput() {
  response_.add_outputs();
}

void InferHandler::InferData::SetOutputDType(size_t i, const std::string &dtype) {
  response_.mutable_outputs(i)->set_dtype(dtype);
}

void InferHandler::InferData::SetOutputShape(size_t i, const std::vector<int64_t> &shape) {
  for (auto s : shape) {
    response_.mutable_outputs(i)->add_shape(s);
  }
}

void InferHandler::InferData::SetOutputData(size_t i, const char* data, size_t bytes) {
  response_.mutable_outputs(i)->set_data(data, bytes);
}

std::string* InferHandler::InferData::MutableOutputData(size_t i) {
  return response_.mutable_outputs(i)->mutable_data();
}

bool InferHandler::InferData::Process(bool ok) {
  switch (status_)
  {
  case Status::kCreate: {
      new InferData{id_ + 1, name_, service_, cq_};
      VLOG(1) << "[Process InferData] [" << GetModelName() << ", Id " << id_ << "]";
      status_ = Status::kFinish;
      
      auto model = ModelInferStore::Get()->GetModel(GetModelName());
      if (!model) {
        LOG(WARNING) << "[Process InferData] Model " << GetModelName() << " not found";
        response_.set_result("model not found");
        responder_.Finish(response_, grpc::Status::CANCELLED, (void*)this);
      } else {
        model->AddJob(this);
      }
      return true;
    }
  case Status::kFinish:
    delete this;
    return true;
  default:
    return false;
  }
}

void TrainHandler::Start() {
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, 2);
  thread_.reset(new std::thread([this, &barrier]() {
    pthread_barrier_wait(&barrier);
    TrainData* train_data = new TrainData{0, "TrainRequest", service_, cq_};

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      train_data = static_cast<TrainData*>(tag);
      CHECK(ok) << "TrainHandler get CallData failed";
      if (!train_data->Process(ok)) {
        delete train_data;
      }
    }
  }));
  pthread_barrier_wait(&barrier);
  LOG(INFO) << "TrainHandler start";
}

void TrainHandler::Stop() {
  thread_->join();
  LOG(INFO) << "TrainHandler stop";
}

bool TrainHandler::TrainData::Process(bool ok) {
    switch (status_)
  {
  case Status::kCreate:
    new TrainData{id_ + 1, name_, service_, cq_};
    LOG(INFO) << "Process TrainData [" << GetModelName() << ", Id " << id_ << "]";
    status_ = Status::kFinish;
    // ModelInferStore::Get()->GetModel("dummy")->AddJob(this);
    // ModelInferStore::Get()->GetModel(GetModelName())->AddJob(this);
    ModelTrainStore::Get()->AddJob(this);
    return true;
  case Status::kFinish:
    // LOG(INFO) << "Process TrainData delete " << std::hex << this;
    delete this;
    return true;
  default:
    return false;
  }
  return true;
}

void TrainHandler::TrainData::SetResult(const std::string &result) {
  response_.set_result(result);
}


} // namespace network
} // namespace colserve