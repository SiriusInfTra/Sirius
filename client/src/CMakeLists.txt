cmake_minimum_required(VERSION 3.20)
project(sirius-client LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
find_package(CUDAToolkit 11 REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

if (NOT DEFINED SIRIUS_ROOT_DIR)
  message(FATAL_ERROR "SIRIUS_ROOT_DIR is not defined")
endif()
if (NOT DEFINED SIRIUS_BUILD_DIR)
  message(FATAL_ERROR "SIRIUS_BUILD_DIR is not defined")
endif()
if (NOT DEFINED TRITON_3RD_DIR)
  message(FATAL_ERROR "TRITON_3RD_DIR is not defined")
endif()
if (NOT DEFINED TRITON_INSTALL_DIR)
  message(FATAL_ERROR "TRITON_INSTALL_DIR is not defined")
endif()

# dep
include(${SIRIUS_ROOT_DIR}/cmake/glog.cmake)
include(${SIRIUS_ROOT_DIR}/cmake/boost.cmake)

include(FetchContent)
FetchContent_Declare(
  CLI11
  SOURCE_DIR ${SIRIUS_ROOT_DIR}/third_party/cli11
)
FetchContent_MakeAvailable(CLI11)

message(">> absl paths: ${TRITON_3RD_DIR}/absl/lib/cmake/absl")
message(">> grpc paths: ${TRITON_3RD_DIR}/grpc/lib/cmake/grpc")
message(">> protobuf paths: ${TRITON_3RD_DIR}/protobuf/lib/cmake/protobuf")

find_package(absl CONFIG REQUIRED NO_DEFAULT_PATH
  PATHS ${TRITON_3RD_DIR}/absl/lib/cmake/absl)
find_package(gRPC CONFIG REQUIRED NODEFAULT_PATH
  PATHS ${TRITON_3RD_DIR}/grpc/lib/cmake/grpc)
find_package(Protobuf REQUIRED CONFIG NO_DEFAULT_PATH 
  PATHS ${TRITON_3RD_DIR}/protobuf/lib/cmake/protobuf)

find_package(TritonCommon REQUIRED NO_DEFAULT_PATH 
  PATHS ${TRITON_INSTALL_DIR}/lib/cmake/TritonCommon)

find_package(TritonClient REQUIRED NO_DEFAULT_PATH 
  PATHS ${TRITON_INSTALL_DIR}/lib/cmake/TritonClient)

# proto
add_library(sirius_proto ${SIRIUS_ROOT_DIR}/proto/colserve.proto)
target_link_libraries(sirius_proto PUBLIC gRPC::grpc++)
set(PROTO_BINARY_DIR ${CMAKE_BINARY_DIR}/generated_proto)
set(PROTO_IMPORT_DIRS ${SIRIUS_ROOT_DIR}/proto/)
file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})
target_include_directories(sirius_proto PUBLIC ${PROTO_BINARY_DIR})

# message(status ">> PROTO_BINARY_DIR: ${PROTO_BINARY_DIR}")
# message(status ">> PROTO_IMPORT_DIRS: ${PROTO_IMPORT_DIRS}")

protobuf_generate(
  TARGET sirius_proto
  IMPORT_DIRS ${PROTO_IMPORT_DIRS}
  PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

protobuf_generate(
    TARGET sirius_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    IMPORT_DIRS ${PROTO_IMPORT_DIRS}
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")


set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(workload_launcher 
  workload_launcher.cc
  workload/sirius_impl.cc
  workload/triton_impl.cc
  workload/warm_cache.cc
  workload/util.cc
)
target_link_libraries(workload_launcher PUBLIC
  sirius_proto
  TritonClient::grpcclient_static
  glog::glog 
  CLI11::CLI11
  Boost::json
)

add_executable(busy_loop_launcher 
  busy_loop_launcher.cc
  workload/sirius_impl.cc
  workload/triton_impl.cc
  workload/warm_cache.cc
  workload/util.cc
)
target_link_libraries(busy_loop_launcher PUBLIC
  sirius_proto
  TritonClient::grpcclient_static
  glog::glog 
  CLI11::CLI11
  Boost::json
)


install(TARGETS workload_launcher DESTINATION ${SIRIUS_BUILD_DIR}/client)
install(TARGETS busy_loop_launcher DESTINATION ${SIRIUS_BUILD_DIR}/client)