cmake_minimum_required(VERSION 3.20)
project(colserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT DEFINED CONDA_PREFIX)
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
endif()
message(STATUS "CONDA_PREFIX: " ${CONDA_PREFIX})

if (NOT DEFINED TVM_HOME)
  set(TVM_HOME $ENV{TVM_HOME})
endif()
message(STATUS "TVM_HOME: " ${TVM_HOME})


# find cuda toolkit
find_package(CUDAToolkit 11 REQUIRED)

# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()


include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  gRPC
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc
)
# FetchContent_MakeAvailable(gRPC)

FetchContent_GetProperties(gRPC)
if (NOT grpc_POPULATED)
  FetchContent_Populate(gRPC)
  # https://github.com/protocolbuffers/protobuf/issues/12185#issuecomment-1594685860
  set(ABSL_ENABLE_INSTALL ON)
  set(BUILD_TYPE_OLD ${CMAKE_BUILD_TYPE})
  set(CMAKE_BUILD_TYPE "Release" CACHE INTERNAL "build type" FORCE)
  add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(CMAKE_BUILD_TYPE ${BUILD_TYPE_OLD} CACHE INTERNAL "build type" FORCE)
endif()

include(cmake/glog.cmake)
# find_package(glog REQUIRED)
# add_definitions(-DDMLC_USE_GLOG)

FetchContent_Declare(
  CLI11
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cli11
)
FetchContent_MakeAvailable(CLI11)


find_package(Torch REQUIRED)
# find cudnn
if((DEFINED CONDA_PREFIX) AND (EXISTS ${CONDA_PREFIX}/include/cudnn.h))
  set(CUDNN_INCLUDE_DIR ${CONDA_PREFIX}/include)
  set(CUDNN_LIBRARY ${CONDA_PREFIX}/lib/libcudnn.so)
  message(STATUS "Found cudnn in ${CONDA_PREFIX}")
elseif(EXISTS /usr/local/cuda/include/cudnn.h)
  set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
  set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so)
  message(STATUS "Found cudnn in /usr/lcoal/cuda")
elseif(EXISTS /usr/include/cudnn.h)
  set(CUDNN_INCLUDE_DIR /usr/include)
  set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/libcudnn.so)
  message(STATUS "Found cudnn in /usr")
else()
  message(STATUS "cudnn not found")
endif()

# find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
# find_program(_PROTOBUF_PROTOC protoc)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)


SET(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/colserve.proto)
set(proto_srcs ${CMAKE_CURRENT_BINARY_DIR}/colserve.pb.cc)
set(proto_hdrs ${CMAKE_CURRENT_BINARY_DIR}/colserve.pb.h)
set(grpc_srcs ${CMAKE_CURRENT_BINARY_DIR}/colserve.grpc.pb.cc)
set(grpc_hdrs ${CMAKE_CURRENT_BINARY_DIR}/colserve.grpc.pb.h)
add_custom_command(
    OUTPUT ${proto_srcs} ${proto_hdrs} ${grpc_srcs} ${grpc_hdrs}
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}/proto"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/proto/colserve.proto"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/proto/colserve.proto"
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(proto ${proto_srcs} ${proto_hdrs} ${grpc_srcs} ${grpc_hdrs})
target_link_libraries(proto ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} )

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/mpool)

add_subdirectory(common)
add_subdirectory(server)

# client
add_executable(client client/client.cc)
target_link_libraries(client grpc++ proto glog::glog CLI11::CLI11)

add_executable(workload_launcher 
  client/workload_launcher.cc
  client/workload/workload.cc
  client/workload/util.cc
)
target_link_libraries(workload_launcher grpc++ proto glog::glog CLI11::CLI11)

add_executable(busy_loop_launcher 
  client/busy_loop_launcher.cc
  client/workload/workload.cc
  client/workload/util.cc
)
target_link_libraries(busy_loop_launcher grpc++ proto glog::glog CLI11::CLI11)


include(cmake/xsched.cmake)

# add_subdirectory(python)
add_subdirectory(pytorch)
