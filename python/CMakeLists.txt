cmake_minimum_required(VERSION 3.24)

project(torch_col LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
set(CMAKE_CXX_STANDARD 17)


# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()


add_library(torch_col SHARED 
  torch_col/csrc/tensor_impl.cc
  torch_col/csrc/aten_type.cc
  torch_col/csrc/dlpack_convert.cc
  torch_col/csrc/control_stub.cc
  torch_col/csrc/init.cc
)

target_include_directories(torch_col PUBLIC sta)
target_include_directories(torch_col PRIVATE colserve)
target_include_directories(torch_col PRIVATE "../")

target_link_libraries(torch_col PRIVATE ${TORCH_LIBRARIES} Python::Python)
target_link_libraries(torch_col PUBLIC sta glog::glog)

# build python ext
add_custom_target(
  torch_col_py_ext ALL
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
  COMMAND ${CONDA_PREFIX}/bin/python
  ARGS setup.py build_ext install 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS 
    torch_col sta
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/torch_col.pxd"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/torch_col.pyx"
)