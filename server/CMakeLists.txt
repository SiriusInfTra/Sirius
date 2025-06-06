cmake_minimum_required(VERSION 3.20)

find_package(Python REQUIRED COMPONENTS Development)

set(torch_python_lib)
foreach(torch_lib IN ITEMS ${TORCH_LIBRARIES})
  if(torch_lib MATCHES ".*torch/lib/libc10.so")
    string(REPLACE "torch/lib/libc10.so" 
                   "torch/lib/libtorch_python.so" 
                   torch_python_lib ${torch_lib})
  endif()
endforeach()

if (NOT torch_python_lib)
  message(FATAL_ERROR "torch_python_lib not found")
endif()

add_executable(${PROJECT_NAME}
  main.cc
  schedule/job_queue.cc
  model_store/infer_model_store.cc
  model_store/infer_model.cc
  model_store/model_cache.cc
  train_launcher.cc
  train_adjuster.cc
  control/controller.cc
  resource_manager.cc
  profiler.cc
  config.cc

  grpc/grpc_server.cc
  tvm/graph.cc
  tvm/executor.cc
  
  llm/llm.cc
  llm/llm_torch_allocator_plugin.cc
  llm/kv_cache_pool.cc
  llm/llm_util.cc
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

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${Python_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} 
  grpc++ proto tvm_runtime CLI11::CLI11 glog::glog
  CUDA::nvml CUDA::cudart CUDA::cuda_driver dcgm
  sta Python::Python Boost::python
  ${TORCH_LIBRARIES} ${torch_python_lib}
)

if (DEFINED USE_NVML_V3)
  target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NVML_V3=${USE_NVML_V3})
  message(STATUS "${PROJECT_NAME} USE_NVML_V3 = ${USE_NVML_V3}")
endif() 