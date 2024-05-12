cmake_minimum_required(VERSION 3.20)

add_library(sta SHARED
  cuda_allocator.cc
  generic_allocator.cc
  torch_allocator.cc
  tvm_allocator.cc
  generic_allocator.h
  mempool.cc
  dtype_helper.cc
  shape_helper.cc
  tensor_methods.cc
  tensor.cc
  init.cc
  sm_partition.cu
  sm_partition.cc
)

target_include_directories(sta PUBLIC "../")
# target_include_directories(sta PUBLIC "../" ${TORCH_INCLUDE_DIRS})
target_link_libraries(sta PUBLIC glog::glog CUDA::cudart CUDA::cuda_driver ${TORCH_LIBRARIES})