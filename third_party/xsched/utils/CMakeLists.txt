file(GLOB_RECURSE UTILS_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
)

add_library(utils SHARED ${UTILS_SRCS})

target_compile_options(utils PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(utils PUBLIC dl)

target_include_directories(utils PUBLIC ${XSCHED_INCLUDE_DIR})

install(TARGETS utils LIBRARY DESTINATION lib COMPONENT xsched)
