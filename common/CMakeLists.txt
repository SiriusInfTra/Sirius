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
  inf_tra_comm/message_queue.cc
  inf_tra_comm/info_board.cc
  inf_tra_comm/communicator.cc
)

target_include_directories(sta PUBLIC 
  "../"
  ${XSCHED_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS})
  
target_link_libraries(sta PUBLIC 
  pages_pool 
  allocator 
  glog::glog 
  CUDA::cudart 
  CUDA::cuda_driver 
  CUDA::nvml
  ${TORCH_LIBRARIES}
  ${XSCHED_LIBRARIES}
  ${Boost_LIBRARIES}
)

