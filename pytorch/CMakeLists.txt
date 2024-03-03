cmake_minimum_required(VERSION 3.20)

project(new_torch_col LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Development)

find_package(Python REQUIRED COMPONENTS Development)

add_library(new_torch_col SHARED 
  csrc/cuda_allocator_plugin.cc
)

target_include_directories(new_torch_col PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(new_torch_col PUBLIC glog::glog ${TORCH_LIBRARIES} Python::Python)