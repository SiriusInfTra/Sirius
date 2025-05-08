file(GLOB_RECURSE SHIM_ASCEND_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(shimascend SHARED
    ${SHIM_ASCEND_SRCS}
    ${SHIM_COMMON_SRCS}
)

target_include_directories(shimascend PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(shimascend PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)

target_link_libraries(shimascend PUBLIC
    utils
    pthread
    preempt
    halascend
    ${SHIM_COMMON_LIBS}
)

target_link_options(shimascend PRIVATE
    -Wl,--exclude-libs,ALL
)

install(TARGETS shimascend LIBRARY DESTINATION lib COMPONENT xsched)
