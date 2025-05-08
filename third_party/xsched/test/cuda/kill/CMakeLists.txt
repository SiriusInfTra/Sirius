find_package(CUDA REQUIRED)

enable_language(CUDA)

file(GLOB_RECURSE CUDA_KILL_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

cuda_add_executable(cuda_kill ${CUDA_KILL_SRCS})

set_target_properties(cuda_kill PROPERTIES CUDA_ARCHITECTURES "${CUDA_GEN_CODE}")

target_include_directories(cuda_kill PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(cuda_kill
    ${CUDA_LIBRARIES}
    utils
    shimcuda
)

install(TARGETS cuda_kill RUNTIME DESTINATION bin)
