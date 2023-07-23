#include <iostream>
#include <grpcpp/grpcpp.h>
#include <chrono>

#include "colserve.grpc.pb.h"

class Client {
 public:
  Client(std::shared_ptr<grpc::Channel> channel)
      : stub_(ColServe::NewStub(channel)) {};

  std::string Hello() {
    EmptyRequest request;
    ServerStatus server_status;

    grpc::ClientContext context;
    grpc::Status status = stub_->GetServerStatus(&context, request, &server_status);

    if (status.ok()) {
      return server_status.status();
    } else {
      std::cerr << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC fail";
    }
  }

  std::string DummyInfer() {
    InferRequest request;
    InferResult infer_result;

    request.set_model("dummy");
    request.set_input("dummy_input");
    
    grpc::ClientContext context;
    grpc::Status status = stub_->Inference(&context, request, &infer_result);

    if (status.ok()) {
      return infer_result.result();
    } else {
      std::cerr << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC fail";
    }
  }
  
  std::string Infer() {
    InferRequest request;
    InferResult infer_result;

    auto input = std::vector<float>(224 * 224 * 3, 1.0);

    request.set_model("resnet50");
    request.set_input_shape(0, 1);
    request.set_input_shape(1, 3);
    request.set_input_shape(2, 224);
    request.set_input_shape(3, 224);
    request.set_input(input.data(), input.size() * sizeof(float));

    grpc::ClientContext context;
    grpc::Status status = stub_->Inference(&context, request, &infer_result);

    if (status.ok()) {
      return infer_result.result();
    } else {
      std::cerr << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC fail";
    }
  }

 private:
  std::unique_ptr<ColServe::Stub> stub_;
};

int main() {
  std::string target = "localhost:8080";
  Client client(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));

  
  auto begin = std::chrono::steady_clock::now();
  auto reply = client.Hello();
  auto end = std::chrono::steady_clock::now();
  std::cout << "recv: " << reply <<  " " 
            << std::chrono::duration<double, std::milli>(end - begin).count()
            << std::endl;

  std::cout << client.DummyInfer() << std::endl;

  begin = std::chrono::steady_clock::now();
  reply = client.Infer();
  end = std::chrono::steady_clock::now();
  std::cout << "recv: " << reply << " "
            << std::chrono::duration<double, std::milli>(end - begin).count()
            << std::endl;

  return 0;
}