// #include <iostream>
// #include <grpcpp/grpcpp.h>

// #include "colserve.grpc.pb.h"

// class Client {
//  public:
//   Client(std::shared_ptr<grpc::Channel> channel)
//       : stub_(ColServe::NewStub(channel)) {};

//   std::string Hello() {
//     EmptyRequest request;
//     ServerStatus server_status;

//     grpc::ClientContext context;
//     grpc::Status status = stub_->GetServerStatus(&context, request, &server_status);

//     if (status.ok()) {
//       return server_status.status();
//     } else {
//       std::cerr << status.error_code() << ": " << status.error_message()
//                 << std::endl;
//       return "RPC fail";
//     }
//   }

//  private:
//   std::unique_ptr<ColServe::Stub> stub_;
// };

// int main() {
//   std::string target = "localhost:8080";
//   Client client(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
//   auto reply = client.Hello();
//   std::cout << "recv: " << reply << std::endl;
//   return 0;
// }

#include <iostream>
#include <grpcpp/grpcpp.h>

#include "colserve.grpc.pb.h"

class Client {
 public:
  Client(std::shared_ptr<grpc::Channel> channel)
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

 private:
  std::unique_ptr<ColServe::Stub> stub_;
};

int main() {
  std::string target = "localhost:8080";
  Client client(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
  auto reply = client.Hello();
  std::cout << "recv: " << reply << std::endl;
  return 0;
}