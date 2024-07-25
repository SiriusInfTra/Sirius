set(VERBOSE OFF)
set(DEVICE_VPI OFF)
set(DEVICE_CUDA ON)
set(CUDA_GEN_CODE 70)
set(DEVICE_CUDLA OFF)
set(DEVICE_ASCEND OFF)
set(XDAG_SUPPORT OFF)

# set(XSCHED_BUILD_TYPE "Release")
# set(XSCHED_BUILD_TYPE "Debug")
set(XSCHED_BUILD_TYPE ${CMAKE_BUILD_TYPE})

set(_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE "${XSCHED_BUILD_TYPE}" CACHE INTERNAL "build type" FORCE)
add_subdirectory(third_party/xsched)
set(CMAKE_BUILD_TYPE ${_CMAKE_BUILD_TYPE} CACHE INTERNAL "build type" FORCE)

add_custom_target(xsched_lib ALL
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} 
                           --component xsched
                           --prefix ${CMAKE_BINARY_DIR}/xsched
)
add_dependencies(xsched_lib utils sched preempt shimcuda)

set(XSCHED_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/)
set(XSCHED_LIBRARIES utils sched preempt shimcuda)