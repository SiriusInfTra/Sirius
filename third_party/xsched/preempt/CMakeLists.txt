file(GLOB_RECURSE PREEMPT_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(preempt SHARED ${PREEMPT_SRCS})

target_include_directories(preempt PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(preempt PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(preempt PUBLIC
    utils
    pthread
)

install(TARGETS preempt LIBRARY DESTINATION lib COMPONENT xsched)
