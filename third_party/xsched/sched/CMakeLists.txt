file(GLOB_RECURSE SCHED_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(sched SHARED ${SCHED_SRCS})

target_include_directories(sched PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(sched PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(sched PUBLIC
    utils
    pthread
    ipc
)

install(TARGETS sched LIBRARY DESTINATION lib COMPONENT xsched)
