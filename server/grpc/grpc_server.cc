#include <thread>
#include <pthread.h>

#include "../model_store.h"
#include "grcp_server.h"


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

DLDataType InferHandler::InferData::GetInputDType() {
  if (request_.input_dtype() == "float32") {
    return DLDataType{kDLFloat, 32, 1};
  } else if (request_.input_dtype() == "int8") {
    return DLDataType{kDLInt, 8, 1};
  } else {
    LOG(FATAL) << "InferData: " << "Unknown input dtype " << request_.input_dtype();
  }
}

std::vector<int64_t> InferHandler::InferData::GetInputShape() {
  std::vector<int64_t> shape;
  for (size_t i = 0; i < request_.input_shape_size(); ++i) {
    shape.push_back(request_.input_shape(i));
  }
  return shape;
}

bool InferHandler::InferData::Process(bool ok) {
  switch (status_)
  {
  case Status::kCreate:
    new InferData{id_ + 1, name_, service_, cq_};
    LOG(INFO) << "Process InferData [" << GetModelName() << ", Id " << id_ << "]";
    status_ = Status::kFinish;
    // ModelStore::Get()->GetModel("dummy")->AddJob(this);
    ModelStore::Get()->GetModel(GetModelName())->AddJob(this);
    return true;
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
  // TODO implement TrainData::Process
  return false;
}


} // namespace network
} // namespace colserve