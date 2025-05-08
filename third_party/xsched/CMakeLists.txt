cmake_minimum_required(VERSION 3.12)

project(XSched)

option(DEVICE_VPI    "Enable support for NVIDIA VPI devices."    OFF)
option(DEVICE_CUDA   "Enable support for NVIDIA CUDA devices."   ON)
option(DEVICE_CUDLA  "Enable support for NVIDIA CUDLA devices."  OFF)
option(DEVICE_ASCEND "Enable support for HUAWEI ASCEND devices." OFF)
option(XDAG_SUPPORT  "Enable support for XDAG framework."        OFF)

set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb -DXSCHED_DEBG")
set(CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DXSCHED_DEBG")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set(XSCHED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(3rdparty)
add_subdirectory(utils)
add_subdirectory(sched)
add_subdirectory(preempt)
add_subdirectory(hal)
add_subdirectory(shim)
# add_subdirectory(service)
# add_subdirectory(test)

# install(DIRECTORY ${XSCHED_INCLUDE_DIR}/ DESTINATION include/xsched COMPONENT xsched)
