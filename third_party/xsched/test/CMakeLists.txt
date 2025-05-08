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
