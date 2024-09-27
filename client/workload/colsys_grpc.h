#ifndef COLSYS_COLSYS_GRPC_H_
#define COLSYS_COLSYS_GRPC_H_

#include <colserve.grpc.pb.h>

using InferClientContext = grpc::ClientContext;
using ServeStub = ColServe::Stub;

inline void SetGPTRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("INT64");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(64);
    request.mutable_inputs(0)->set_data(data);
}

inline void SetBertRequest(InferRequest &request, const std::string &model, const std::string &data, const std::string &mask) {
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("INT64");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(64);
    request.mutable_inputs(0)->set_data(data);
    request.add_inputs();
    request.mutable_inputs(1)->set_dtype("INT64");
    request.mutable_inputs(1)->add_shape(1);
    request.mutable_inputs(1)->add_shape(64);
    request.mutable_inputs(1)->set_data(mask);
}

inline void SetResnetRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("FP32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(3);
    request.mutable_inputs(0)->add_shape(224);
    request.mutable_inputs(0)->add_shape(224);
    request.mutable_inputs(0)->set_data(data);
}

inline void SetInceptionRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.set_model(model);
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("FP32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(3);
    request.mutable_inputs(0)->add_shape(299);
    request.mutable_inputs(0)->add_shape(299);
    request.mutable_inputs(0)->set_data(data);
}



#endif