cmake_minimum_required(VERSION 3.24)
project(colserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# find cuda toolkit
find_package(CUDAToolkit 11 REQUIRED)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.56.2  # e.g v1.28.0
)
# FetchContent_MakeAvailable(gRPC)
FetchContent_GetProperties(gRPC)
if (NOT grpc_POPULATED)
  FetchContent_Populate(gRPC)
  set(BUILD_TYPE_OLD ${CMAKE_BUILD_TYPE})
  set(CMAKE_BUILD_TYPE "Release" CACHE INTERNAL "build type" FORCE)
  add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR})
  set(CMAKE_BUILD_TYPE ${BUILD_TYPE_OLD} CACHE INTERNAL "build type" FORCE)
endif()

FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog
  GIT_TAG        v0.6.0
)
# FetchContent_MakeAvailable(glog)
FetchContent_GetProperties(glog)
if (NOT glog_POPULATED)
  FetchContent_Populate(glog)
  set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS ON CACHE INTERNAL "build shared library" FORCE)
  add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR})
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_OLD} CACHE INTERNAL "library build type" FORCE)
endif()

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG        v2.3.2
)
FetchContent_MakeAvailable(CLI11)

find_package(Torch REQUIRED)
# find cudnn
set(CUDNN_INCLUDE_DIR $ENV{CONDA_PREFIX}/include)
set(CUDNN_LIBRARY $ENV{CONDA_PREFIX}/lib/libcudnn.so)

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


# add tvm runtime
include_directories(
  ${TVM_HOME}/include
  ${TVM_HOME}/3rdparty/dmlc-core/include
  ${TVM_HOME}/3rdparty/dlpack/include
)

link_directories(
  ${TVM_HOME}/build
  ${TVM_HOME}/3rdparty/dmlc-core/build
)

# sta lib
add_library(sta SHARED
  server/sta/cuda_allocator.cc
  server/sta/tensor_pool.cc
  server/sta/tensor_methods.cc
  server/sta/shape_helper.cc
  server/sta/dtype_helper.cc
  server/sta/init.cc
)
# target_include_directories(sta INTERFACE ${TVM_HOME}/3rdparty/dlpack/include)
target_include_directories(sta PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/server/)
target_link_libraries(sta PRIVATE CUDA::cudart glog::glog
                          PUBLIC ${TORCH_LIBRARIES})

add_executable(${PROJECT_NAME}
  server/main.cc
  server/grpc/grpc_server.cc
  server/tvm/graph_executor_factory.cc
  server/tvm/graph_executor.cc
  server/job_queue.cc
  server/model_infer_store.cc
  server/model_train_store.cc
  server/controller.cc
  server/profiler.cc
  server/config.cc
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

target_link_libraries(${PROJECT_NAME} 
  grpc++ proto tvm_runtime CLI11::CLI11 glog::glog
  CUDA::nvml CUDA::cudart CUDA::cuda_driver
  sta
)

add_executable(client client/client.cc)
target_link_libraries(client grpc++ proto glog::glog CLI11::CLI11)

add_executable(hybrid_workload 
  client/hybrid_workload.cc
  client/workload/workload.cc
  client/workload/util.cc
)
target_link_libraries(hybrid_workload grpc++ proto glog::glog CLI11::CLI11)


add_subdirectory(python)