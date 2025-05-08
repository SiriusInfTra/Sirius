file(GLOB_RECURSE HAL_VPI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(halvpi SHARED
    ${HAL_VPI_SRCS}
)

target_include_directories(halvpi PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(halvpi PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(halvpi PUBLIC
    utils
    pthread
    preempt
)

install(TARGETS halvpi LIBRARY DESTINATION lib COMPONENT xsched)
