file(GLOB_RECURSE SHIM_CUDLA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(shimcudla SHARED
    ${SHIM_CUDLA_SRCS}
    ${SHIM_COMMON_SRCS}
)

target_include_directories(shimcudla PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(shimcudla PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)

target_link_libraries(shimcudla PUBLIC
    utils
    pthread
    preempt
    halcudla
    ${SHIM_COMMON_LIBS}
)

target_link_options(shimcudla PRIVATE
    -Wl,--exclude-libs,ALL
)

install(TARGETS shimcudla LIBRARY DESTINATION lib COMPONENT xsched)
