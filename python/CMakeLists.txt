cmake_minimum_required(VERSION 3.24)

project(torch_col LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)


find_package(Python REQUIRED COMPONENTS Development)

FetchContent_Declare(
  cudnn_frontend
  GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
  GIT_TAG        v0.6.2  # e.g v1.28.0
)
# FetchContent_MakeAvailable(cudnn_frontend)

FetchContent_GetProperties(cudnn_frontend)
if (NOT cudnn_frontend_POPULATED)
  FetchContent_Populate(cudnn_frontend)
  set(CUDNN_FRONTEND_INCLUDE_DIR ${cudnn_frontend_SOURCE_DIR}/include)
endif()


# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

add_library(torch_col_tensor SHARED
  torch_col/csrc/tensor_impl.cc
  torch_col/csrc/convolution.cc
  torch_col/csrc/cudnn/cudnn_custom.cc
  torch_col/csrc/cudnn/ConvShared.cpp
  torch_col/csrc/cudnn/Conv_v7.cpp
  torch_col/csrc/cudnn/Conv_v8.cpp
  torch_col/csrc/cudnn/ConvPlaceholders.cpp
  torch_col/csrc/cudnn/Descriptors.cpp
  torch_col/csrc/cudnn/Types.cpp
  torch_col/csrc/override_ops/nonzero.cu
  torch_col/csrc/aten_type.cc
  torch_col/csrc/dlpack_convert.cc
)

target_compile_options(torch_col_tensor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -D__CUDA_NO_HALF_OPERATORS__>)

target_compile_definitions(torch_col_tensor PRIVATE -DHAVE_AVX2_CPU_DEFINITION)
target_compile_definitions(torch_col_tensor PRIVATE -DHAVE_AVX512_CPU_DEFINITION)
target_compile_definitions(torch_col_tensor PRIVATE -DUSE_EXPERIMENTAL_CUDNN_V8_API)

target_include_directories(torch_col_tensor PUBLIC sta)
target_include_directories(torch_col_tensor PRIVATE colserve)
target_include_directories(torch_col_tensor PRIVATE "../")
target_include_directories(torch_col_tensor PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(torch_col_tensor PRIVATE ${CUDNN_FRONTEND_INCLUDE_DIR})


target_link_libraries(torch_col_tensor PRIVATE ${TORCH_LIBRARIES} Python::Python)
target_link_libraries(torch_col_tensor PUBLIC sta glog::glog)

add_library(torch_col SHARED 
  torch_col/csrc/control_stub.cc
  torch_col/csrc/init.cc
)

target_include_directories(torch_col PRIVATE "../")
target_link_libraries(torch_col PRIVATE glog::glog sta)


# build python ext
add_custom_target(
  torch_col_py_ext ALL
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
  COMMAND $ENV{CONDA_PREFIX}/bin/python
  ARGS setup.py build_ext install 
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS 
    torch_col sta torch_col_tensor
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/torch_col.pxd"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/torch_col.pyx"
)