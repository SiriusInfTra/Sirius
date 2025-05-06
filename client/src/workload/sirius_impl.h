#ifndef SIRIUS_SIRIUS_IMPL_H_
#define SIRIUS_SIRIUS_IMPL_H_

#include <colserve.grpc.pb.h>
#include <grpcpp/channel.h>
#include <glog/logging.h>
#include <boost/json.hpp>
#include "util.h"

#define SIRIUS_CLIENT_IMPL_NAMESPACE sirius_backend
#define __SIRIUS_CLIENT_BACKEND__ __SIRIUS_BACKEND__

namespace colserve::workload::SIRIUS_CLIENT_IMPL_NAMESPACE {

using TrainRequest = sirius::TrainRequest;
using TrainResult = sirius::TrainResult;
using EmptyRequest = sirius::EmptyRequest;
using EmptyResult = sirius::EmptyResult;

using InferRequest = sirius::InferRequest;
using InferResult = sirius::InferResult;

using ServerStatus = sirius::ServerStatus;
using ServeStub = sirius::ColServe::Stub;
using InferWorkloadDoneRequest = sirius::InferWorkloadDoneRequest;
using InferenceWorkloadStartRequest = sirius::InferenceWorkloadStartRequest;

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
                          const std::string &data,
                          int input_length, int output_length) {
  request.set_model(model);
  request.add_inputs();
  request.mutable_inputs(0)->set_dtype("char");
  request.mutable_inputs(0)->add_shape(data.size());
  request.mutable_inputs(0)->set_data(data);

  boost::json::object json_obj;
  json_obj["max_tokens"] = output_length;
  std::string json_str = boost::json::serialize(json_obj);

  request.add_inputs();
  request.mutable_inputs(1)->set_dtype("char"); 
  request.mutable_inputs(1)->add_shape(json_str.size());
  request.mutable_inputs(1)->set_data(json_str);
}

inline std::unique_ptr<ServeStub> NewStub(std::shared_ptr<grpc::Channel> channel) {
  return sirius::ColServe::NewStub(channel);
}

inline void StubAsyncInferenceDone(ServeStub &stub, ::grpc::ClientContext *context, 
                                   const std::string &model_name) {
  // only triton backend
}


}

#endif