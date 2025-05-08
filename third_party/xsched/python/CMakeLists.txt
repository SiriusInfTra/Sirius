cmake_minimum_required(VERSION 3.20)
project(PySched LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} SHARED src/PySched.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME}
    PUBLIC middleware 
    PRIVATE scheduler xpreempt)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH $ORIGIN)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include COMPONENT xsched)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib COMPONENT xsched)