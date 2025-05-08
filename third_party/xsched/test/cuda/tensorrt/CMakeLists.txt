list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

file(GLOB_RECURSE TRT_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(trt_common OBJECT ${TRT_COMMON_SRCS})

target_include_directories(trt_common PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

file(GLOB_RECURSE TRT_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)

foreach(TRT_MAIN_SRC ${TRT_MAIN_SRCS})

    get_filename_component(TRT_TARGET ${TRT_MAIN_SRC} NAME_WE)

    set(TRT_TARGET trt_${TRT_TARGET})

    add_executable(${TRT_TARGET}
        ${TRT_MAIN_SRC}
        $<TARGET_OBJECTS:trt_common>
    )

    target_include_directories(${TRT_TARGET} PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

    target_link_libraries(${TRT_TARGET} PRIVATE
        ${CUDA_LIBRARIES}
        TensorRT::nvinfer
        TensorRT::nvinfer_plugin
        TensorRT::nvonnxparser
        utils
        shimcuda
    )

    install(TARGETS ${TRT_TARGET} RUNTIME DESTINATION bin)

endforeach()
