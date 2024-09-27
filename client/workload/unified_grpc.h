#ifndef COLSYS_UNIFIED_GRPC_H_
#define COLSYS_UNIFIED_GRPC_H_

#include <cstdint>
#include <grpc_service.pb.h>
#include <grpcpp/channel.h>
#include <grpcpp/support/status.h>
#include <memory>
#include <cstring>
#include <colserve.grpc.pb.h>

#include <grpc_client.h>
#include <grpc_service.grpc.pb.h>
#include <glog/logging.h>

using TrainRequest = colsys::TrainRequest;
using TrainResult = colsys::TrainResult;



// using InferWorkloadDone
using EmptyRequest = inference::ServerLiveRequest;
using EmptyResult = inference::ServerLiveResponse;


struct InferenceWorkloadStartRequest {
    int64_t time_stamp_;
    double delay_before_profile_;

    void set_time_stamp(int64_t time_stamp) {
        time_stamp_ = time_stamp;
    }

    void set_delay_before_profile(double delay_before_profile) {
        delay_before_profile_ = delay_before_profile;
    }
};
struct InferWorkloadDoneRequest {
    int64_t time_stamp_;

    void set_time_stamp(int64_t time_stamp) {
        time_stamp_ = time_stamp;
    }
};



struct ServerStatus {
    inference::ServerLiveResponse valaue;

    std::string status() {
        return valaue.live() ? "live" : "dead";
    }
    
};


struct InferRequest {
    inference::ModelInferRequest value;

    const std::string& model() const {
        return value.model_name();
    }
};



inline void SetGPTRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.value.set_model_name(model);
    // request.req.set_raw_input_contents()
    auto* input = request.value.add_inputs();
    input->set_name("input0");
    input->set_datatype("INT64");
    input->add_shape(1);
    input->add_shape(64);
    
    auto contents = input->mutable_contents();
    for (int k = 0; k < 64; ++k) {
        contents->add_int64_contents(reinterpret_cast<const int64_t*>(data.data())[k]);
    }
    
}

inline void SetBertRequest(InferRequest &request, const std::string &model, const std::string &ids, const std::string &mask) {
    request.value.set_model_name(model);
    auto* input = request.value.add_inputs();
    input->set_name("input0");
    input->set_datatype("INT64");
    input->add_shape(1);
    input->add_shape(64);
    
    auto contents = input->mutable_contents();
    for (int k = 0; k < 64; ++k) {
        contents->add_int64_contents( reinterpret_cast<const int64_t*>(ids.data())[k]);
    }
    
    auto* input1 = request.value.add_inputs();
    input1->set_name("input1");
    input1->set_datatype("INT64");
    input1->add_shape(1);
    input1->add_shape(64);
    
    auto contents1 = input1->mutable_contents();
    for (int k = 0; k < 64; ++k) {
        contents1->add_int64_contents(reinterpret_cast<const int64_t*>(mask.data())[k]);
    }
}

inline void SetMnistRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.value.set_model_name(model);
    auto* input = request.value.add_inputs();
    input->set_name("input0");
    input->set_datatype("FP32");
    input->add_shape(1);
    input->add_shape(28);
    input->add_shape(28);

    auto contents = input->mutable_contents();
    for (int k = 0; k < 28 * 28; ++k) {
        contents->add_fp32_contents(reinterpret_cast<const float*>(data.data())[k]);
    }
}
inline void SetResnetRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.value.set_model_name(model);
    auto* input = request.value.add_inputs();
    input->set_name("input0");
    input->set_datatype("FP32");
    input->add_shape(1);
    input->add_shape(3);
    input->add_shape(224);
    input->add_shape(224);

    auto contents = input->mutable_contents();
    for (int k = 0; k < 3 * 224 * 224; ++k) {
        contents->add_fp32_contents(reinterpret_cast<const float*>(data.data())[k]);
    }
}

inline void SetInceptionRequest(InferRequest &request, const std::string &model, const std::string &data) {
    request.value.set_model_name(model);
    auto* input = request.value.add_inputs();
    input->set_name("input0");
    input->set_datatype("FP32");
    input->add_shape(1);
    input->add_shape(3);
    input->add_shape(299);
    input->add_shape(299);

    auto contents = input->mutable_contents();
    // contents->ParseFromString(data);
    for (int k = 0; k < 3 * 299 * 299; ++k) {
        contents->add_fp32_contents(reinterpret_cast<const float*>(data.data())[k]);
    }
}


class TensorData {
private:
    std::string raw_data_;
    std::string dtype_;
public:
    TensorData(const void * ptr, size_t numel, const std::string &dtype) 
        : raw_data_(reinterpret_cast<const char*>(ptr), numel), 
            dtype_(dtype) {}

    std::string &data() {
        return raw_data_;
    }

    const std::string &dtype() const {
        return dtype_;
    }
};

struct InferResult {
    inference::ModelInferResponse value;

    TensorData outputs(int index) const {
        auto output =  value.outputs(0);
        auto &datatype = output.datatype();
        if (datatype == "FP32") {
            return TensorData(
                output.contents().fp32_contents().data(), 
                output.contents().fp32_contents_size() * sizeof(float), 
                datatype);
        } else if (datatype == "INT64") {
            return TensorData(
                output.contents().int64_contents().data(), 
                output.contents().int64_contents_size() * sizeof(float), 
                datatype);
        } else {
            LOG(FATAL) << "Unsupported datatype: " << datatype;
        }
    }

    const std::string &result() const {
        return outputs(0).data();
    }
};


struct AsyncInferResult {
    using PendingInferResult = 
        grpc::ClientAsyncResponseReader<inference::ModelInferResponse>;
    std::unique_ptr<PendingInferResult> value;

    AsyncInferResult(std::unique_ptr<PendingInferResult> result) : value(std::move(result)) {

    }

    void Finish(InferResult *result, grpc::Status* status, void* tag) {
        value->Finish(&result->value, status, tag);
    }
};

struct AsyncServerStatus {
    using PendingServerStatus = 
        grpc::ClientAsyncResponseReader<inference::ServerLiveResponse>;
    std::unique_ptr<PendingServerStatus> value;

    AsyncServerStatus(std::unique_ptr<PendingServerStatus> result) : value(std::move(result)) {

    }

    void Finish(ServerStatus *result, grpc::Status* status, void* tag) {
        value->Finish(&result->valaue, status, tag);
    }
};

class ServeStub {
private:
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;
public:
    ServeStub(std::shared_ptr<grpc::Channel> channel) {
        stub_ = inference::GRPCInferenceService::NewStub(channel);
    }

    grpc::Status Inference(grpc::ClientContext* context, const InferRequest& request, InferResult* response) {
        inference::ModelInferRequest req;
        stub_->ModelInfer(context, request.value, &response->value);
        return grpc::Status::OK;
    }

    grpc::Status Train(::grpc::ClientContext* context, const TrainRequest& request, TrainResult* response) {
        return grpc::Status::OK;
    }

    std::unique_ptr<AsyncInferResult> AsyncInference(::grpc::ClientContext* context, const ::InferRequest& request, ::grpc::CompletionQueue* cq) {
        auto result = stub_->AsyncModelInfer(context, request.value, cq);
        return std::make_unique<AsyncInferResult>(std::move(result));
    }

    grpc::Status GetServerStatus(::grpc::ClientContext* context, const EmptyRequest &request, ServerStatus* response) {
        return stub_->ServerLive(context, request, &response->valaue);
    }

    std::unique_ptr<AsyncServerStatus> AsyncGetServerStatus(::grpc::ClientContext* context, const EmptyRequest &request, ::grpc::CompletionQueue* cq) {
        auto result = stub_->AsyncServerLive(context, request, cq);
        return std::make_unique<AsyncServerStatus>(std::move(result));
    }

    grpc::Status WarmupDone(::grpc::ClientContext* context, const EmptyRequest &request, EmptyResult* response) {
        return grpc::Status::OK; // Triton don't need warmup done!
    }

    grpc::Status InferenceWorkloadStart(::grpc::ClientContext* context, const InferenceWorkloadStartRequest &request, EmptyResult* response) {
        return grpc::Status::OK;
    }

    grpc::Status InferenceWorkloadDone(::grpc::ClientContext* context, const InferWorkloadDoneRequest &request, EmptyResult* response) {
        return grpc::Status::OK;
    }



};

inline std::unique_ptr<ServeStub> NewStub(std::shared_ptr<grpc::Channel> channel) {
    return std::make_unique<ServeStub>(channel);
}
#endif