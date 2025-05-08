file(GLOB_RECURSE HAL_CUDLA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(halcudla SHARED
    ${HAL_CUDLA_SRCS}
)

target_include_directories(halcudla PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(halcudla PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(halcudla PUBLIC
    utils
    pthread
    preempt
)

install(TARGETS halcudla LIBRARY DESTINATION lib COMPONENT xsched)
