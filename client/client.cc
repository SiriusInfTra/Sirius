#include <iostream>
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <fstream>
#include <glog/logging.h>

#include "colserve.grpc.pb.h"

std::string ReadInput(const std::string &data_path) {
  std::ifstream data_file{data_path, std::ios::binary};
  CHECK(data_file.good()) << "data " << data_path << " not exist";
  std::string data{std::istreambuf_iterator<char>(data_file), std::istreambuf_iterator<char>()};
  data_file.close();
  return data;
}

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
    request.add_inputs();
    request.mutable_inputs(0)->set_data("dummy_input");
    
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
  
  std::string InferMnist(const std::string &data) {
    InferRequest request;
    InferResult infer_result;

    // auto input = std::vector<float>(224 * 224 * 3, 1.0);
    

    request.set_model("mnist");
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("float32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(28);
    request.mutable_inputs(0)->add_shape(28);
    request.mutable_inputs(0)->set_data(data.data(), data.size() * sizeof(char));

    LOG(INFO) << "set input done";

    grpc::ClientContext context;
    grpc::Status status = stub_->Inference(&context, request, &infer_result);

    if (status.ok()) {
      std::stringstream ss;
      auto output = reinterpret_cast<const float*>(infer_result.outputs(0).data().data());
      ss << "[";
      float prob = -std::numeric_limits<float>::max();
      int number = -1;
      for (int i = 0; i < 10; i++) {
        ss << output[i] << " ";
        if (output[i] > prob) {
          prob = output[i];
          number = i;
        }
      }
      ss << "] ";
      ss << infer_result.outputs(0).dtype();
      ss << ", num is " << number;
      return ss.str();
    } else {
      std::cerr << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC fail";
    }
  }

  std::string InferResnet(const std::string &data) {
    InferRequest request;
    InferResult infer_result;

    request.set_model("resnet152");
    request.add_inputs();
    request.mutable_inputs(0)->set_dtype("float32");
    request.mutable_inputs(0)->add_shape(1);
    request.mutable_inputs(0)->add_shape(3);
    request.mutable_inputs(0)->add_shape(224);
    request.mutable_inputs(0)->add_shape(224);
    request.mutable_inputs(0)->set_data(data.data(), data.size());

    grpc::ClientContext context;
    grpc::Status status = stub_->Inference(&context, request, &infer_result);

    if (status.ok()) {
      std::stringstream ss;
      auto output = reinterpret_cast<const float*>(infer_result.outputs(0).data().data());
      std::vector<int> label(1000);
      std::iota(label.begin(), label.end(), 0);
      std::sort(label.begin(), label.end(), [&](int x, int y) {
        return output[x] > output[y];
      });
      for (size_t i = 0; i < 5; i++) {
        ss << "(" << label[i] << ":" << output[label[i]] << ") ";
      }
      return ss.str();
    } else {
      std::cerr << status.error_code() << ": " << status.error_message()
                << std::endl;
      return "RPC fail";
    }
  }

  std::string Train() {
    TrainRequest request;
    TrainResult train_result;
    
    request.set_model("resnet152");
    request.set_args("batch-size=1, num-epoch=1");
    
    grpc::ClientContext context;
    grpc::Status status = stub_->Train(&context, request, &train_result);

    if (status.ok()) {
      return train_result.result();
    } else {
      std::cerr << status.error_code() << ": " << status.error_message()
                << std::endl;
      return train_result.result();
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

  auto mnist_data = ReadInput("data/mnist/input-0.bin");
  auto resnet_data = ReadInput("data/resnet/input-0.bin");

  begin = std::chrono::steady_clock::now();
  reply = client.InferMnist(mnist_data);
  end = std::chrono::steady_clock::now();
  std::cout << "mnist recv: " << reply << " "
            << std::chrono::duration<double, std::milli>(end - begin).count()
            << std::endl;

  // begin = std::chrono::steady_clock::now();
  // reply = client.InferResnet(resnet_data);
  // end = std::chrono::steady_clock::now();
  // std::cout << "resnet recv: " << reply << " "
  //           << std::chrono::duration<double, std::milli>(end - begin).count()
  //           << std::endl;

  std::cout << client.Train() << std::endl;

  return 0;
}