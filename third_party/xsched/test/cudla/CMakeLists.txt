find_package(CUDA REQUIRED)

set(CUDA_SEARCH_PATHS
    PATHS "/usr" "/usr/local" "/usr/local/cuda"
)
set(CUDA_SEARCH_PATH_SUFFIXES lib lib64)

find_library(CUDLA_LIBRARY NAMES cudla ${CUDA_SEARCH_PATHS} PATH_SUFFIXES ${CUDA_SEARCH_PATH_SUFFIXES})
find_library(CUDART_LIBRARY NAMES cudart ${CUDA_SEARCH_PATHS} PATH_SUFFIXES ${CUDA_SEARCH_PATH_SUFFIXES})

file(GLOB_RECURSE CUDLA_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(cudla_common OBJECT ${CUDLA_COMMON_SRCS})

target_include_directories(cudla_common PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(cudla_common PRIVATE
    -Wall
    -Wextra
    -Werror
)

file(GLOB_RECURSE CUDLA_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)

foreach(CUDLA_MAIN_SRC ${CUDLA_MAIN_SRCS})

    get_filename_component(CUDLA_TARGET ${CUDLA_MAIN_SRC} NAME_WE)

    set(CUDLA_TARGET cudla_${CUDLA_TARGET})

    add_executable(${CUDLA_TARGET}
        ${CUDLA_MAIN_SRC}
        $<TARGET_OBJECTS:cudla_common>
    )

    target_include_directories(${CUDLA_TARGET} PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

    target_compile_options(${CUDLA_TARGET} PRIVATE
        -Wall
        -Wextra
        -Werror
    )

    target_link_libraries(${CUDLA_TARGET} PRIVATE
        ${CUDLA_LIBRARY}
        ${CUDART_LIBRARY}
        utils
        shimcudla
    )

    install(TARGETS ${CUDLA_TARGET} RUNTIME DESTINATION bin)

endforeach()
