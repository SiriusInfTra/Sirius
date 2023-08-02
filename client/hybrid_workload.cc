#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <thread>
#include <grpcpp/grpcpp.h>
#include <glog/logging.h>

#include "colserve.grpc.pb.h"

std::string ReadInput(const std::string &data_path) {
  std::ifstream data_file{data_path, std::ios::binary};
  CHECK(data_file.good()) << "data " << data_path << " not exist";
  std::string data{std::istreambuf_iterator<char>(data_file), std::istreambuf_iterator<char>()};
  data_file.close();
  return data;
}

class Workload {
 public:
  Workload(std::shared_ptr<grpc::Channel> channel)
      : stub_(ColServe::NewStub(channel)) {};

  std::string Hello() {
    EmptyRequest request;
    ServerStatus server_status;

    grpc::ClientContext context;
    grpc::CompletionQueue cq;
    grpc::Status status;

    std::unique_ptr<grpc::ClientAsyncResponseReader<ServerStatus>> rpc(
        stub_->AsyncGetServerStatus(&context, request, &cq));
    rpc->Finish(&server_status, &status, (void*)1);

    void *tag;
    bool ok = false;
    cq.Next(&tag, &ok);
    if (status.ok()) {
      return server_status.status();
    } else {
      return "RPC failed";
    }
  }

  void Mnist(size_t concurrency) {
    static std::vector<std::string> mnist_input_datas;
    if (mnist_input_datas.empty()) {
      
    }
  }

  void Resnet(size_t concurrency) {

  }
  
  void Train() {
    
  }

 private:
  std::atomic<bool> running_{false};
  std::vector<std::unique_ptr<std::thread>> threads_;

  std::unique_ptr<ColServe::Stub> stub_;
};

int main() {
  std::string target = "localhost:8080";
  Workload workload(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
  auto reply = workload.Hello();
  std::cout << "recv: " << reply << std::endl;
  return 0;
}