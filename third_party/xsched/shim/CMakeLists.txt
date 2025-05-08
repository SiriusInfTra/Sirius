file(GLOB_RECURSE SHIM_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/common/*.c
    ${CMAKE_CURRENT_LIST_DIR}/common/*.cpp
)

set(SHIM_COMMON_LIBS
    sched
)

if(DEVICE_VPI)
    add_subdirectory(vpi)
endif()

if(DEVICE_CUDA)
    add_subdirectory(cuda)
endif()

if(DEVICE_CUDLA)
    add_subdirectory(cudla)
endif()

if(DEVICE_ASCEND)
    add_subdirectory(ascend)
endif()

if(XDAG_SUPPORT)
    add_subdirectory(xdag)
endif()
