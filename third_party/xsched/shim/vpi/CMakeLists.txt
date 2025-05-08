file(GLOB_RECURSE SHIM_VPI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(shimvpi SHARED
    ${SHIM_VPI_SRCS}
    ${SHIM_COMMON_SRCS}
)

target_include_directories(shimvpi PUBLIC ${XSCHED_INCLUDE_DIR})

target_compile_options(shimvpi PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
    -fvisibility=hidden
)

target_link_libraries(shimvpi PUBLIC
    utils
    pthread
    preempt
    halvpi
    ${SHIM_COMMON_LIBS}
)

target_link_options(shimvpi PRIVATE
    -Wl,--exclude-libs,ALL
)

install(TARGETS shimvpi LIBRARY DESTINATION lib COMPONENT xsched)
