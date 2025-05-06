cmake_minimum_required(VERSION 3.20)
project(colserve LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# ===============================================================
# find dependencies & build subdirectories
# ===============================================================

find_package(CUDAToolkit 11 REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES is set to ${CMAKE_CUDA_ARCHITECTURES}")

FetchContent_Declare(
  CLI11
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cli11
)
FetchContent_MakeAvailable(CLI11)
find_package(Torch REQUIRED)


include(cmake/cudnn.cmake)
include(cmake/boost.cmake)
include(cmake/glog.cmake)
include(cmake/grpc.cmake)
include(cmake/xsched.cmake)


add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/mpool)
add_subdirectory(common)
add_subdirectory(server)
add_subdirectory(pytorch)

# ===============================================================
# building client
# ===============================================================

set(client_cmake_extra_args)
if (DEFINED Boost_ROOT)
  list(APPEND client_cmake_extra_args -DBoost_ROOT=${Boost_ROOT})
endif()

include(ExternalProject)
ExternalProject_Add(client
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/client
    PREFIX ${CMAKE_BINARY_DIR}/client
    CMAKE_ARGS 
      -DSIRIUS_ROOT_DIR=${CMAKE_SOURCE_DIR}
      -DSIRIUS_BUILD_DIR=${CMAKE_BINARY_DIR}
      ${client_cmake_extra_args}
    CMAKE_CACHE_ARGS
      -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
    INSTALL_COMMAND ""
    USES_TERMINAL_BUILD true
    BUILD_ALWAYS ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
)

# ===============================================================
# link release/debug build directory to ./build
# ===============================================================

if (NOT ${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR}/build)
  if (EXISTS "${CMAKE_SOURCE_DIR}/build")
      execute_process(COMMAND unlink ${CMAKE_SOURCE_DIR}/build)
  endif()
  if (NOT EXISTS "${CMAKE_SOURCE_DIR}/build")
      execute_process(COMMAND ln -s ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build)
  endif()
endif()