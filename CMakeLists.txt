cmake_minimum_required(VERSION 3.20)
project(colserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT DEFINED CONDA_PREFIX)
  set(CONDA_PREFIX $ENV{CONDA_PREFIX})
endif()
message(STATUS "CONDA_PREFIX: " ${CONDA_PREFIX})

if (NOT DEFINED TVM_HOME)
  set(TVM_HOME $ENV{TVM_HOME})
endif()
message(STATUS "TVM_HOME: " ${TVM_HOME})


# find cuda toolkit
find_package(CUDAToolkit 11 REQUIRED)

# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()


# add dependencies
# find cli11
FetchContent_Declare(
  CLI11
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cli11
)
FetchContent_MakeAvailable(CLI11)


# find torch and cudnn
find_package(Torch REQUIRED)
if((DEFINED CONDA_PREFIX) AND (EXISTS ${CONDA_PREFIX}/include/cudnn.h))
  set(CUDNN_INCLUDE_DIR ${CONDA_PREFIX}/include)
  set(CUDNN_LIBRARY ${CONDA_PREFIX}/lib/libcudnn.so)
  message(STATUS "Found cudnn in ${CONDA_PREFIX}")
elseif(EXISTS /usr/local/cuda/include/cudnn.h)
  set(CUDNN_INCLUDE_DIR /usr/local/cuda/include)
  set(CUDNN_LIBRARY /usr/local/cuda/lib64/libcudnn.so)
  message(STATUS "Found cudnn in /usr/lcoal/cuda")
elseif(EXISTS /usr/include/cudnn.h)
  set(CUDNN_INCLUDE_DIR /usr/include)
  set(CUDNN_LIBRARY /usr/lib/x86_64-linux-gnu/libcudnn.so)
  message(STATUS "Found cudnn in /usr")
else()
  message(STATUS "cudnn not found")
endif()

include(cmake/boost.cmake)
include(cmake/glog.cmake)
include(cmake/grpc.cmake)
include(cmake/xsched.cmake)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/mpool)


# building
add_subdirectory(common)
add_subdirectory(server)
add_subdirectory(pytorch)


include(ExternalProject)
ExternalProject_Add(client
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/client
    CMAKE_ARGS 
      -DCOLSYS_ROOT_DIR=${CMAKE_SOURCE_DIR}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    INSTALL_COMMAND ""
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
)

# link release/debug build directory to ./build
if (NOT ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR}/build)
  if (EXISTS "${CMAKE_SOURCE_DIR}/build")
      execute_process(COMMAND unlink ${CMAKE_SOURCE_DIR}/build)
  endif()
  if (NOT EXISTS "${CMAKE_SOURCE_DIR}/build")
      execute_process(COMMAND ln -s ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build)
  endif()
endif()