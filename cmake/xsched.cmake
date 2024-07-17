set(VERBOSE OFF)
set(DEVICE_VPI OFF)
set(DEVICE_CUDA ON)
set(CUDA_GEN_CODE 70)
set(DEVICE_CUDLA OFF)
set(DEVICE_ASCEND OFF)
set(XDAG_SUPPORT OFF)

# set(XSCHED_BUILD_TYPE "Release")
set(XSCHED_BUILD_TYPE "Debug")

# set(_CMAKE_GENERATOR ${CMAKE_GENERATOR})
set(_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE "${XSCHED_BUILD_TYPE}" CACHE INTERNAL "build type" FORCE)
# set(CMAKE_GENERATOR "Unix Makefiles")
add_subdirectory(third_party/xsched)
# set(CMAKE_GENERATOR ${_CMAKE_GENERATOR})
set(CMAKE_BUILD_TYPE ${_CMAKE_BUILD_TYPE} CACHE INTERNAL "build type" FORCE)

add_custom_target(xsched_lib ALL
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --component xsched
    --prefix ${CMAKE_BINARY_DIR}/xsched
  # DEPENDS 
    # ${CMAKE_BINARY_DIR}/third_party/xsched/python/libPySched.so
    # ${CMAKE_BINARY_DIR}/third_party/xsched/
    # ${CMAKE_BINARY_DIR}/third_party/xsched/middleware/libmiddleware.so
    # ${CMAKE_BINARY_DIR}/third_party/xsched/scheduler/libscheduler.so
    # ${CMAKE_BINARY_DIR}/third_party/xsched/xpreempt/libxpreempt.so
)
add_dependencies(xsched_lib utils sched preempt shimcuda)

set(XSCHED_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/)
set(XSCHED_LIBRARY_DIR ${CMAKE_BINARY_DIR}/xsched/lib)
set(XSCHED_LIBRARIES shimcuda)