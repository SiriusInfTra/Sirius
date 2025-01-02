#ifndef COLSYS_COLSYS_IMPL_H_
#define COLSYS_COLSYS_IMPL_H_

#include <colserve.grpc.pb.h>
#include <grpcpp/channel.h>
#include <glog/logging.h>

#define COLSYS_CLIENT_IMPL_NAMESPACE colsys_backend
namespace colserve::workload::COLSYS_CLIENT_IMPL_NAMESPACE {

using TrainRequest = colsys::TrainRequest;
using TrainResult = colsys::TrainResult;
using EmptyRequest = colsys::EmptyRequest;
using EmptyResult = colsys::EmptyResult;

using InferRequest = colsys::InferRequest;
using InferResult = colsys::InferResult;

using ServerStatus = colsys::ServerStatus;
using ServeStub = colsys::ColServe::Stub;
using InferWorkloadDoneRequest = colsys::InferWorkloadDoneRequest;
using InferenceWorkloadStartRequest = colsys::InferenceWorkloadStartRequest;

using AsyncInferResult = grpc::ClientAsyncResponseReader<InferResult>;
using AsyncServerStatus = grpc::ClientAsyncResponseReader<ServerStatus>;


inline void SetGPTRequest(InferRequest &request, const std::string &model, 
                          const std::string &data) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("int64");
  request.mutable_inputs(0)->add_shape(1);
  request.mutable_inputs(0)->add_shape(64);
  request.mutable_inputs(0)->set_data(data);
}

inline void SetBertRequest(InferRequest &request, const std::string &model, 
                           const std::string &data, const std::string &mask) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("int64");
  request.mutable_inputs(0)->add_shape(1);
  request.mutable_inputs(0)->add_shape(64);
  request.mutable_inputs(0)->set_data(data);
  request.add_inputs();
  request.mutable_inputs(1)->set_dtype("int64");
  request.mutable_inputs(1)->add_shape(1);
  request.mutable_inputs(1)->add_shape(64);
  request.mutable_inputs(1)->set_data(mask);
}

inline void SetResnetRequest(InferRequest &request, const std::string &model, 
                             const std::string &data) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("float32");
  request.mutable_inputs(0)->add_shape(1);
  request.mutable_inputs(0)->add_shape(3);
  request.mutable_inputs(0)->add_shape(224);
  request.mutable_inputs(0)->add_shape(224);
  request.mutable_inputs(0)->set_data(data);
}

inline void SetInceptionRequest(InferRequest &request, const std::string &model, 
                                const std::string &data) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("float32");
  request.mutable_inputs(0)->add_shape(1);
  request.mutable_inputs(0)->add_shape(3);
  request.mutable_inputs(0)->add_shape(299);
  request.mutable_inputs(0)->add_shape(299);
  request.mutable_inputs(0)->set_data(data);
}

inline void SetMnistRequest(InferRequest &request, const std::string &model, 
                            const std::string &data) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("float32");
  request.mutable_inputs(0)->add_shape(1);
  request.mutable_inputs(0)->add_shape(28);
  request.mutable_inputs(0)->add_shape(28);
  request.mutable_inputs(0)->set_data(data);
}

inline void SetLLMRequest(InferRequest &request, const std::string &model,
                          const std::string &data) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("char");
  request.mutable_inputs(0)->add_shape(data.size());
  request.mutable_inputs(0)->set_data(data);
}

inline std::unique_ptr<ServeStub> NewStub(std::shared_ptr<grpc::Channel> channel) {
  return colsys::ColServe::NewStub(channel);
}

inline void StubAsyncInferenceDone(ServeStub &stub, ::grpc::ClientContext *context, 
                                   const std::string &model_name) {
  // only triton backend
}


}

#endif