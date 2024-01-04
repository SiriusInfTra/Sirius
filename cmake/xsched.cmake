set(VERBOSE OFF)
set(HOOK_LOG OFF)
set(PLAT_CUDA ON)
set(ENABLE_INSTRUMENT OFF)
set(CUDA_GEN_CODE 70)
set(PLAT_VPI OFF)
set(XSCHED_BUILD_TYPE "Release")

# set(_CMAKE_GENERATOR ${CMAKE_GENERATOR})
set(_CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/xsched)
set(_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/xsched" CACHE INTERNAL "install prefix" FORCE)
set(CMAKE_BUILD_TYPE "${XSCHED_BUILD_TYPE}" CACHE INTERNAL "build type" FORCE)
# set(CMAKE_GENERATOR "Unix Makefiles")
add_subdirectory(third_party/xsched)
# set(CMAKE_GENERATOR ${_CMAKE_GENERATOR})
set(CMAKE_INSTALL_PREFIX ${_CMAKE_INSTALL_PREFIX} CACHE INTERNAL "install prefix" FORCE)
set(CMAKE_BUILD_TYPE ${_CMAKE_BUILD_TYPE} CACHE INTERNAL "build type" FORCE)

add_custom_target(xsched_lib ALL
  COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --component xsched
  DEPENDS 
    ${CMAKE_BINARY_DIR}/third_party/xsched/python/libPySched.so
    ${CMAKE_BINARY_DIR}/third_party/xsched/middleware/libmiddleware.so
    ${CMAKE_BINARY_DIR}/third_party/xsched/scheduler/libscheduler.so
    ${CMAKE_BINARY_DIR}/third_party/xsched/xpreempt/libxpreempt.so
)
add_dependencies(xsched_lib middleware scheduler xpreempt PySched)

# add_custom_command(
#     OUTPUT ${CMAKE_BINARY_DIR}/__xsched_install
#     COMMAND ${CMAKE_COMMAND} -E echo "Updating __xsched_install"
# )

