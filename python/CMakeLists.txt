cmake_minimum_required(VERSION 3.20)

project(torch_col LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)


find_package(Python REQUIRED COMPONENTS Development)

set(cudnn_frontend_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/cudnn_frontend)
FetchContent_Declare(
  cudnn_frontend
  SOURCE_DIR    ${cudnn_frontend_SOURCE_DIR}
)
# FetchContent_MakeAvailable(cudnn_frontend)

FetchContent_GetProperties(cudnn_frontend)
if (NOT cudnn_frontend_POPULATED)
  FetchContent_Populate(cudnn_frontend)
  set(CUDNN_FRONTEND_INCLUDE_DIR ${cudnn_frontend_SOURCE_DIR}/include)
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
  torch_col/csrc/override_ops/shallow_copy.cc
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


target_link_libraries(torch_col_tensor PUBLIC sta glog::glog  ${TORCH_LIBRARIES} Python::Python)

add_library(torch_col SHARED 
  torch_col/csrc/control_stub.cc
  torch_col/csrc/torch_helper.cc
  torch_col/csrc/init.cc
)

target_include_directories(torch_col PRIVATE "../")
target_link_libraries(torch_col PRIVATE torch_col_tensor)


# build python ext
add_custom_target(
  torch_col_py_ext ALL
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
)

if (NOT DEFINED CONDA_PREFIX)
  set(PYTHON_EXECUTABLE /usr/bin/python)
else()
  set(PYTHON_EXECUTABLE ${CONDA_PREFIX}/bin/python)
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
  COMMAND BUILD_DIR=${CMAKE_BINARY_DIR} ${PYTHON_EXECUTABLE}
  ARGS setup.py build_ext install
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS 
    torch_col sta torch_col_tensor
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/torch_col.pxd"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/torch_col.pyx"
)