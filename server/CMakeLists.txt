cmake_minimum_required(VERSION 3.20)

add_executable(${PROJECT_NAME}
  main.cc
  cache.cc
  job_queue.cc
  model_infer_store.cc
  model_train_store.cc
  controller.cc
  profiler.cc
  config.cc

  grpc/grpc_server.cc

  tvm/graph_executor.cc
  tvm/graph_executor_factory.cc
)

# add tvm runtime
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${TVM_HOME}/include 
  ${TVM_HOME}/3rdparty/dmlc-core/include 
  ${TVM_HOME}/3rdparty/dlpack/include
)

target_link_directories(${PROJECT_NAME} PUBLIC
  ${TVM_HOME}/build
  ${TVM_HOME}/3rdparty/dmlc-core/build
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

target_link_libraries(${PROJECT_NAME} 
  grpc++ proto tvm_runtime CLI11::CLI11 glog::glog
  CUDA::nvml CUDA::cudart CUDA::cuda_driver
  sta
)

if (DEFINED USE_NVML_V3)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NVML_V3=${USE_NVML_V3})
  message(STATUS "${PROJECT_NAME} not use nvml v3 api")
endif()