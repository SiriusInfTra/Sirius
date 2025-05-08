add_library(CLI11 INTERFACE)

target_include_directories(CLI11 INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/include
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/CLI DESTINATION include)
