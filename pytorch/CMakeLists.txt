cmake_minimum_required(VERSION 3.20)

project(new_torch_col LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Development)

add_library(new_torch_col SHARED 
  torch_col/csrc/cuda_allocator_plugin.cc
)

target_include_directories(new_torch_col PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(new_torch_col PUBLIC 
  sta
  ${TORCH_LIBRARIES}
  Python::Python
)

set_target_properties(new_torch_col PROPERTIES BUILD_RPATH $ORIGIN)


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

file(GLOB TORCH_COL_PYTHON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/*.py")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/torch_col.egg-info
  COMMAND BUILD_DIR=${CMAKE_BINARY_DIR} ${PYTHON_EXECUTABLE}
  ARGS setup.py install
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS 
    new_torch_col sta
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/memory.pyx"
    ${TORCH_COL_PYTHON_FILES}
)