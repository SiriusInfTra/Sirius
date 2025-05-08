file(GLOB_RECURSE SHIM_CUDA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(shimcuda SHARED
    ${SHIM_CUDA_SRCS}
    ${SHIM_COMMON_SRCS}
)

target_include_directories(shimcuda PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(shimcuda PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)

target_link_libraries(shimcuda PUBLIC
    utils
    pthread
    preempt
    halcuda
    ${SHIM_COMMON_LIBS}
)

target_link_options(shimcuda PRIVATE
    -Wl,--exclude-libs,ALL
)

install(TARGETS shimcuda LIBRARY DESTINATION lib COMPONENT xsched)

set(SHIM_CUDA_SOFTLINKS
    libcuda.so.1
    libcuda.so
)

foreach(SOFTLINK ${SHIM_CUDA_SOFTLINKS})
    install(CODE "
        execute_process(
            COMMAND ln -s libshimcuda.so ${SOFTLINK}
            WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
        )
        message(\"-- Installing: softlink created: ${SOFTLINK}\")
    " COMPONENT xsched)
endforeach()
