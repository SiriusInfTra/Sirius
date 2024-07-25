cmake_minimum_required(VERSION 3.20)

project(new_torch_col LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Development)

add_library(new_torch_col SHARED 
  torch_col/csrc/torch_allocator_plugin.cc
  torch_col/csrc/control_stub.cc
  torch_col/csrc/mem_tagging.cc
  torch_col/csrc/util.cc
  torch_col/csrc/config.cc
  torch_col/csrc/fake_engine.cc
  torch_col/csrc/init.cc
)

add_dependencies(new_torch_col xsched_lib)


target_include_directories(new_torch_col PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(new_torch_col PRIVATE ${CMAKE_SOURCE_DIR}/pytorch)
target_include_directories(new_torch_col PRIVATE ${CMAKE_BINARY_DIR}/xsched/include)
target_link_directories(new_torch_col PRIVATE ${CMAKE_BINARY_DIR}/xsched/lib)

set(torch_python_lib)
foreach(torch_lib IN ITEMS ${TORCH_LIBRARIES})
  if(torch_lib MATCHES ".*torch/lib/libc10.so")
    string(REPLACE "torch/lib/libc10.so" "torch/lib/libtorch_python.so" torch_python_lib ${torch_lib})
  endif()
endforeach()

if (NOT torch_python_lib)
  message(FATAL_ERROR "torch_python_lib not found")
endif()

message("torch lib: ${TORCH_LIBRARIES}, torch_python_lib: ${torch_python_lib}")

target_link_libraries(new_torch_col PRIVATE 
  sta
  ${TORCH_LIBRARIES}
  ${torch_python_lib}
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
  ARGS setup.py develop
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS 
    new_torch_col sta
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/ctrl_stub.pxi"
    "${CMAKE_CURRENT_SOURCE_DIR}/torch_col/main.pyx"
    ${TORCH_COL_PYTHON_FILES}
)