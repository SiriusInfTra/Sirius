find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE VPI_COMMON_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(vpi_common OBJECT ${VPI_COMMON_SRCS})

target_include_directories(vpi_common PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${XSCHED_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(vpi_common PRIVATE
    -Wall
    -Wextra
    -Werror
)

file(GLOB_RECURSE VPI_MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)

foreach(VPI_MAIN_SRC ${VPI_MAIN_SRCS})

    get_filename_component(VPI_TARGET ${VPI_MAIN_SRC} NAME_WE)

    set(VPI_TARGET vpi_${VPI_TARGET})

    add_executable(${VPI_TARGET}
        ${VPI_MAIN_SRC}
        $<TARGET_OBJECTS:vpi_common>
    )

    target_include_directories(${VPI_TARGET} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${XSCHED_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

    target_compile_options(${VPI_TARGET} PRIVATE
        -Wall
        -Wextra
        -Werror
    )

    target_link_libraries(${VPI_TARGET} PRIVATE
        vpi
        pthread
        opencv_core
        opencv_videoio
        opencv_imgproc
        opencv_imgcodecs
        utils
        shimvpi
    )

    install(TARGETS ${VPI_TARGET} RUNTIME DESTINATION bin)

endforeach()
