cmake_minimum_required(VERSION 3.20)
project(colsys-client LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
find_package(CUDAToolkit 11 REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

set(COLSYS_ROOT_DIR ${PROJECT_SOURCE_DIR}/../)
set(TRITON_BUILD_PREFIX ${PROJECT_SOURCE_DIR}/third_party/build/triton-client-prefix/src/triton-client-build)
set(TRITON_INSTALL_DIR ${TRITON_BUILD_PREFIX}/cc-clients/install)
set(TRITON_3RD_DID ${TRITON_BUILD_PREFIX}/third-party)
message("TRITON_BUILD_PREFIX ${TRITON_BUILD_PREFIX}")
message("TRITON_INSTALL_DIR ${TRITON_INSTALL_DIR}")
message("TRITON_3RD_DID ${TRITON_3RD_DID}")
find_package(TritonCommon REQUIRED NO_DEFAULT_PATH 
  PATHS ${TRITON_INSTALL_DIR}/lib/cmake/TritonCommon)

find_package(TritonClient REQUIRED NO_DEFAULT_PATH 
  PATHS ${TRITON_INSTALL_DIR}/lib/cmake/TritonClient)

include(/disk2/wyk/gpu-colocation/cmake/glog.cmake)
include(${TRITON_INSTALL_DIR}/lib/cmake/TritonClient/TritonClientConfig.cmake)


include(${TRITON_3RD_DID}/protobuf/lib/cmake/protobuf/protobuf-config.cmake)


find_package(absl CONFIG REQUIRED NO_DEFAULT_PATH
  PATHS ${TRITON_3RD_DID}/grpc/lib/cmake/absl)
find_package(gRPC CONFIG REQUIRED NODEFAULT_PATH
  PATHS ${TRITON_3RD_DID}/grpc/lib/cmake/grpc)
find_package(Protobuf REQUIRED CONFIG NO_DEFAULT_PATH 
  PATHS ${TRITON_3RD_DID}/protobuf/lib/cmake/protobuf)


include(FetchContent)
FetchContent_Declare(
  CLI11
  SOURCE_DIR /disk2/wyk/gpu-colocation/third_party/cli11
)
FetchContent_MakeAvailable(CLI11)
add_library(colsys_proto ${COLSYS_ROOT_DIR}/proto/colserve.proto)
target_link_libraries(colsys_proto PUBLIC gRPC::grpc++)
set(PROTO_BINARY_DIR ${CMAKE_BINARY_DIR}/generated_proto)
set(PROTO_IMPORT_DIRS ${COLSYS_ROOT_DIR}/proto/)
file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})
target_include_directories(colsys_proto PUBLIC ${PROTO_BINARY_DIR})

protobuf_generate(
  TARGET colsys_proto
  IMPORT_DIRS ${PROTO_IMPORT_DIRS}
  PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

protobuf_generate(
    TARGET colsys_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    IMPORT_DIRS ${PROTO_IMPORT_DIRS}
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


add_executable(workload_launcher 
  workload_launcher.cc
  workload/colsys_impl.cc
  workload/triton_impl.cc
  workload/warm_cache.cc
  workload/util.cc
)
target_link_libraries(workload_launcher PUBLIC
  colsys_proto
  TritonClient::grpcclient_static
  glog::glog 
  CLI11::CLI11
)

add_executable(busy_loop_launcher 
  busy_loop_launcher.cc
  workload/colsys_impl.cc
  workload/triton_impl.cc
  workload/warm_cache.cc
  workload/util.cc
)
target_link_libraries(busy_loop_launcher PUBLIC
  colsys_proto
  TritonClient::grpcclient_static
  glog::glog 
  CLI11::CLI11
)