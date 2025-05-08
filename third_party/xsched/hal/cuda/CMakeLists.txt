if(NOT DEFINED CUDA_GEN_CODE)
    message(FATAL_ERROR "Error: gpu generation code not set! Pass by -DCUDA_GEN_CODE=xx")
endif()

set(ARCH_RELATED_INSTRUMENT_SRC
    ${CMAKE_CURRENT_LIST_DIR}/arch/sm${CUDA_GEN_CODE}.cpp
)

file(GLOB_RECURSE HAL_CUDA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(halcuda SHARED
    ${HAL_CUDA_SRCS}
    ${ARCH_RELATED_INSTRUMENT_SRC}
)

target_include_directories(halcuda PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(halcuda PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(halcuda PUBLIC
    utils
    pthread
    cuxtra
    preempt
)

install(TARGETS halcuda LIBRARY DESTINATION lib COMPONENT xsched)
