file(GLOB_RECURSE XCLI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_executable(xcli ${XCLI_SRCS})

target_include_directories(xcli PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(xcli PRIVATE
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(xcli PUBLIC
    utils
    sched
    ipc
    CLI11
    jsoncpp
    tabulate
)

install(TARGETS xcli RUNTIME DESTINATION bin)
