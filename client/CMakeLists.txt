cmake_minimum_required(VERSION 3.20)
project(colsys-client LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
find_package(CUDAToolkit 11 REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()


set(COLSYS_ROOT_DIR "..")

set(TRITON_INSTALL_DIR ${CMAKE_BINARY_DIR}/triton/install)
set(TRITON_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/triton-client)
message("TRITON_INSTALL_DIR ${TRITON_INSTALL_DIR}")


include(ExternalProject)
set(TRITON_COMMON_REPO_TAG "r23.12")
set(TRITON_THIRD_PARTY_REPO_TAG "r23.12")
set(TRITON_CORE_REPO_TAG "r23.12")

ExternalProject_Add(triton-client
    SOURCE_DIR ${TRITON_SOURCE_DIR}
    CMAKE_ARGS 
        -DCMAKE_INSTALL_PREFIX=${TRITON_INSTALL_DIR}
        -DTRITON_ENABLE_CC_HTTP=OFF 
        -DTRITON_ENABLE_CC_GRPC=ON 
        -DTRITON_ENABLE_PERF_ANALYZER=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_C_API=OFF 
        -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF
        -DTRITON_ENABLE_GPU=ON 
        -DTRITON_ENABLE_EXAMPLES=OFF 
        -DTRITON_ENABLE_TESTS=OFF
        -DTRITON_KEEP_TYPEINFO=ON
        -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG}
        -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG}
        -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG}
    INSTALL_COMMAND ""
)
find_package(TritonCommon REQUIRED NO_DEFAULT_PATH 
  PATHS ${TRITON_INSTALL_DIR}/lib/cmake/TritonCommon)

find_package(TritonClient REQUIRED NO_DEFAULT_PATH 
  PATHS ${TRITON_INSTALL_DIR}/lib/cmake/TritonClient)

include(/disk2/wyk/gpu-colocation/cmake/glog.cmake)
# include(${TRITON_INSTALL_DIR}/lib/cmake/TritonClient/TritonClientConfig.cmake)


# include(${CMAKE_BINARY_DIR}/triton-client-prefix/src/triton-client-build/third-party/protobuf/lib/cmake/protobuf/protobuf-config.cmake)

# find_package(triton_client REQUIRED HINTS .)
# find_package(triton_proto REQUIRED HINTS .)
find_package(absl CONFIG REQUIRED NO_DEFAULT_PATH
  PATHS /disk2/wyk/gpu-colocation/build/client-prefix/src/client-build/triton-client-prefix/src/triton-client-build/third-party/grpc/lib/cmake/absl)
find_package(gRPC CONFIG REQUIRED NODEFAULT_PATH
  PATHS /disk2/wyk/gpu-colocation/build/client-prefix/src/client-build/triton-client-prefix/src/triton-client-build/third-party/grpc/lib/cmake/grpc)
find_package(Protobuf REQUIRED CONFIG NO_DEFAULT_PATH 
  PATHS /disk2/wyk/gpu-colocation/build/client-prefix/src/client-build/triton-client-prefix/src/triton-client-build/third-party/protobuf/lib/cmake/protobuf)


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
message("!!!!!!!!!!!!!!!PROTO_OUTPUT ${PROTO_OUTPUT}")
target_include_directories(colsys_proto PUBLIC ${PROTO_BINARY_DIR})
message("COLSYS_ROOT_DIR ${COLSYS_ROOT_DIR}/proto/colserve.proto")

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

add_executable(client client.cc)
target_link_libraries(client
  colsys_proto
  TritonClient::grpcclient_static
  glog::glog 
  CLI11::CLI11
)

add_executable(workload_launcher 
  workload_launcher.cc
  workload/workload.cc
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
  workload/workload.cc
  workload/util.cc
)
target_link_libraries(busy_loop_launcher PUBLIC
  colsys_proto
  TritonClient::grpcclient_static
  glog::glog 
  CLI11::CLI11
)