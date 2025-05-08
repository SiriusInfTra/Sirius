set(ACL_INCLUDE_DIRS
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/include"
)

set(ACL_LINK_DIRS
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub"
)

file(GLOB_RECURSE ASCEND_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(ascend_common OBJECT ${ASCEND_COMMON_SRCS})

target_include_directories(ascend_common PRIVATE
    ${ACL_INCLUDE_DIRS}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(ascend_common PRIVATE
    -Wall
    -Wextra
    -Werror
    -std=c++11
)

file(GLOB_RECURSE ASCEND_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)

foreach(ASCEND_MAIN_SRC ${ASCEND_MAIN_SRCS})

    get_filename_component(ASCEND_TARGET ${ASCEND_MAIN_SRC} NAME_WE)

    set(ASCEND_TARGET ascend_${ASCEND_TARGET})

    add_executable(${ASCEND_TARGET}
        ${ASCEND_MAIN_SRC}
        $<TARGET_OBJECTS:ascend_common>
    )

    target_include_directories(${ASCEND_TARGET} PRIVATE
        ${ACL_INCLUDE_DIRS}
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

    target_compile_options(${ASCEND_TARGET} PRIVATE
        -Wall
        -Wextra
        -Werror
        -std=c++11
    )

    target_link_directories(${ASCEND_TARGET} PRIVATE
        ${ACL_LINK_DIRS}
    )

    target_link_libraries(${ASCEND_TARGET} PRIVATE
        pthread
        ascendcl
        utils
        shimascend
    )

    install(TARGETS ${ASCEND_TARGET} RUNTIME DESTINATION bin)

endforeach()
