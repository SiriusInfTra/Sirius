file(GLOB_RECURSE SHIM_XDAG_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(shimxdag SHARED
    ${SHIM_XDAG_SRCS}
)

target_include_directories(shimxdag PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(shimxdag PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)

target_link_libraries(shimxdag PUBLIC
    utils
    pthread
    cuxtra
    ${SHIM_COMMON_LIBS}
)

target_link_options(shimxdag PRIVATE
    -Wl,--exclude-libs,ALL
)

set(SHIM_CUDLA_SOFTLINKS
    libcuda.so.1
    libcuda.so
    libcudla.so.1
    libcudla.so
)

install(TARGETS shimxdag LIBRARY DESTINATION lib)

foreach(SOFTLINK ${SHIM_CUDLA_SOFTLINKS})
    install(CODE "
        execute_process(
            COMMAND ln -s libshimxdag.so ${SOFTLINK}
            WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib
        )
        message(\"-- Installing: softlink created: ${SOFTLINK}\")
    ")
endforeach()
