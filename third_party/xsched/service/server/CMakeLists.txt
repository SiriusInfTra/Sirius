file(GLOB_RECURSE XSERVER_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_executable(xserver ${XSERVER_SRCS})

target_include_directories(xserver PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(xserver PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(xserver PUBLIC
    utils
    sched
    ipc
    jsoncpp
)

install(TARGETS xserver RUNTIME DESTINATION bin)
