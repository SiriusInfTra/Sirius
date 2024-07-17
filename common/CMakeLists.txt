cmake_minimum_required(VERSION 3.20)

add_library(sta SHARED
  cuda_allocator.cc
  dtype_helper.cc
  shape_helper.cc
  tensor_methods.cc
  tensor.cc
  device_manager.cc
  sm_partition.cu
  sm_partition.cc
  xsched_ctrl.cc
)

target_include_directories(sta PUBLIC 
  "../"
  ${XSCHED_INCLUDE_DIR})
target_link_directories(sta PUBLIC ${XSCHED_LIBRARY_DIR})
# target_include_directories(sta PUBLIC "../" ${TORCH_INCLUDE_DIRS})
target_link_libraries(sta PUBLIC 
  pages_pool 
  allocator 
  glog::glog 
  CUDA::cudart 
  CUDA::cuda_driver 
  CUDA::nvml
  ${TORCH_LIBRARIES}
  ${XSCHED_LIBRARIES})

