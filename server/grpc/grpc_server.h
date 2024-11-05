#ifndef COLSERVE_GRPC_SERVER_H
#define COLSERVE_GRPC_SERVER_H

// #include <dlpack/dlpack.h>
#include <colserve.grpc.pb.h>

#include <server/logging_as_glog.h>
#include <common/tensor/dlpack.h>

#include <grpc++/grpc++.h>
#include <thread>

namespace colserve {
namespace network {
using namespace colsys;
class CommonHandler;
class InferHandler;
class TrainHandler;

class GRPCServer {
  
 public:
  void Start(const std::string &addr);
  void Stop();

  ~GRPCServer();

  class Handler {
   public:
    Handler(ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
        : service_(service), cq_(cq) {}

   protected:    
    ColServe::AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
  };

  class CallData {
   public:
    CallData(uint64_t id, const std::string &name, 
             ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
        : id_(id), name_(name), service_(service), cq_(cq) {}
    virtual bool Process(bool ok) = 0;
    uint64_t GetId() { return id_; }
    const std::string& GetName() { return name_; }
   protected:
    const uint64_t id_;
    const std::string name_;
    ColServe::AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
  };
 private:

  std::unique_ptr<grpc::ServerCompletionQueue> common_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> infer_cq_;
  std::unique_ptr<grpc::ServerCompletionQueue> train_cq_;

  std::unique_ptr<CommonHandler> common_handler_;
  std::unique_ptr<InferHandler> infer_handler_;
  std::unique_ptr<TrainHandler> train_handler_;

  ColServe::AsyncService service_;
  std::unique_ptr<grpc::Server> server_;
  
};


class CommonHandler : public GRPCServer::Handler { 
 public:
  CommonHandler(ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
      : Handler(service, cq) {}

  void Start();
  void Stop();

  template <typename RequestType, typename ResponseType>
  class CommonData : public GRPCServer::CallData {
   public:
    using RegisterFunc = std::function<void(CommonData<RequestType, ResponseType>*)>;
    using ExecuteFunc = std::function<void(CommonData<RequestType, ResponseType>*)>;

    CommonData(uint64_t id, const std::string &name, 
               ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq,
               RegisterFunc register_func, ExecuteFunc execute_func)
        : GRPCServer::CallData(id, name, service, cq), responder_(&ctx_),
          register_func_(register_func), execute_func_(execute_func),
          status_(Status::kCreate) {
      register_func_(this);
    }
    bool Process(bool ok) override {
      switch(status_) {
        case Status::kCreate: {
          new CommonData<RequestType, ResponseType>{
              id_ + 1, name_, service_, cq_, register_func_, execute_func_};
          status_ = Status::kFinish;
          LOG(INFO) << "Process CommonData [" << name_ << ", Id " << id_ << "]";
          execute_func_(this);
          return true;
        }
        case Status::kFinish: {
          delete this;
          return true;   
        }
        default: return false;
      }
    }
    enum class Status { kCreate, kFinish };
    friend class CommonHandler;
   private:
    RequestType request_;
    ResponseType response_;
    grpc::ServerContext ctx_;
    grpc::ServerAsyncResponseWriter<ResponseType> responder_;

    RegisterFunc register_func_;
    ExecuteFunc execute_func_;

    Status status_;
  };

 private:
  std::unique_ptr<std::thread> thread_;
  void SetupCallData();
};


class InferHandler : public GRPCServer::Handler {
 public:
  InferHandler(ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
      : Handler(service, cq) {}

  void Start();
  void Stop();

  class InferData : public GRPCServer::CallData {
   public:
    InferData(uint64_t id, const std::string &name,
              ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
        : CallData(id, name, service, cq), responder_(&ctx_),
          status_(Status::kCreate) {
      service_->RequestInference(&ctx_, &request_, &responder_, 
                    cq_, cq_, (void*)this);
    }
    
    bool Process(bool ok) override;
    const std::string& GetModelName() { return request_.model(); }
    DLDataType GetInputDType(size_t i);
    std::vector<int64_t> GetInputShape(size_t i);
    const char* GetInputData(size_t i);
    size_t GetInputBytes(size_t i);

    void AddOuput();
    void SetOutputDType(size_t i, const std::string &dtype);
    void SetOutputShape(size_t i, const std::vector<int64_t> &shape);
    void SetOutputData(size_t i, const char* data, size_t bytes);
    std::string* MutableOutputData(size_t i);

    InferResult& GetResponse() { return response_; }
    grpc::ServerAsyncResponseWriter<InferResult>& GetResponder() { return responder_; }

    enum class Status { kCreate, kFinish };
    friend class InferHandler;
   private:
    InferRequest request_;
    InferResult response_;
    grpc::ServerContext ctx_;
    grpc::ServerAsyncResponseWriter<InferResult> responder_;

    Status status_;
  };

 private:
  std::unique_ptr<std::thread> thread_;
};


class TrainHandler : public GRPCServer::Handler {
 public:
  TrainHandler(ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
      : Handler(service, cq) {}

  void Start();
  void Stop();

  class TrainData : public GRPCServer::CallData {
   public:
    TrainData(uint64_t id, const std::string &name,
              ColServe::AsyncService* service, grpc::ServerCompletionQueue* cq)
        : GRPCServer::CallData(id, name, service, cq), responder_(&ctx_),
          status_(Status::kCreate) {
      service_->RequestTrain(&ctx_, &request_, &responder_, 
                    cq_, cq_, (void*)this);
    }

    bool Process(bool ok) override;
    const std::string& GetModelName() { return request_.model(); }
    // <Key>=<Value>, ...
    const std::string& GetTrainArgs() { return request_.args(); }

    void SetResult(const std::string& result);

    TrainResult& GetResponse() { return response_; }
    grpc::ServerAsyncResponseWriter<TrainResult>& GetResponder() { return responder_; }
    
    enum class Status { kCreate, kFinish };
    friend class TrainHandler;
    
   private:
    TrainRequest request_;
    TrainResult response_;
    grpc::ServerContext ctx_;
    grpc::ServerAsyncResponseWriter<TrainResult> responder_;

    Status status_;
  };

 private:
  std::unique_ptr<std::thread> thread_;
};

} // namespace network
} // namespace colserve

#endif