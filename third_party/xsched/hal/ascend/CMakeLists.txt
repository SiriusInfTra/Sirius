file(GLOB_RECURSE HAL_ASCEND_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(halascend SHARED
    ${HAL_ASCEND_SRCS}
)

target_include_directories(halascend PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(halascend PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(halascend PUBLIC
    utils
    pthread
    preempt
)

install(TARGETS halascend LIBRARY DESTINATION lib COMPONENT xsched)
