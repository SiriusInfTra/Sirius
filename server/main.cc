#include <iostream>
#include "model_store.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <glog/logging.h>

#include "grpc/grcp_server.h"
#include "colserve.grpc.pb.h"


int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  colserve::ModelStore::Init("models");

  std::string server_address("0.0.0.0:8080");
  colserve::network::GRPCServer server;
  server.Start(server_address);

  server.Stop();
  LOG(INFO) << "server has shotdown";
}