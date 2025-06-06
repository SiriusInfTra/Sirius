cmake_minimum_required(VERSION 3.20)
project(sirius-client LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
find_package(CUDAToolkit 11 REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

if (NOT DEFINED SIRIUS_ROOT_DIR)
  message(FATAL_ERROR "SIRIUS_ROOT_DIR is not defined")
endif()
if (NOT DEFINED SIRIUS_BUILD_DIR)
  message(FATAL_ERROR "SIRIUS_BUILD_DIR is not defined")
endif()

set(client_impl_cmake_extra_args)
if (DEFINED Boost_ROOT)
    list(APPEND client_impl_cmake_extra_args -DBoost_ROOT=${Boost_ROOT})
    message(STATUS ">>> Client: Use Boost: ${Boost_ROOT}")
endif()

set(TRITON_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/triton-client)
set(TRITON_PREFIX ${SIRIUS_BUILD_DIR}/client/triton-client)
set(TRITON_BUILD_DIR ${TRITON_PREFIX}/src/triton-client-build)

# get triton version
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${TRITON_SOURCE_DIR}
    OUTPUT_VARIABLE TRITON_SHA1
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND git "branch" "-r" "--sort=-committerdate" "--contains" ${TRITON_SHA1}
    COMMAND sed -nE "s/.*origin\\/([^ ]*).*/\\1/p"
    WORKING_DIRECTORY ${TRITON_SOURCE_DIR}
    OUTPUT_VARIABLE TRITON_BRANCH_NAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


set(TRITON_CC_CLIENT_BUILD_DIR ${TRITON_BUILD_DIR}/cc-clients)

include(ExternalProject)
ExternalProject_Add(triton-client
    SOURCE_DIR ${TRITON_SOURCE_DIR}
    PREFIX ${TRITON_PREFIX}
    CMAKE_ARGS 
        -DTRITON_ENABLE_CC_HTTP=OFF 
        -DTRITON_ENABLE_CC_GRPC=ON 
        -DTRITON_ENABLE_PERF_ANALYZER=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_C_API=OFF 
        -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF
        -DTRITON_ENABLE_GPU=ON 
        -DTRITON_ENABLE_EXAMPLES=OFF 
        -DTRITON_ENABLE_TESTS=OFF
        -DTRITON_COMMON_REPO_TAG=${TRITON_BRANCH_NAME}
        -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_BRANCH_NAME}
        -DTRITON_CORE_REPO_TAG=${TRITON_BRANCH_NAME}
        -DCMAKE_INSTALL_PREFIX=${TRITON_BUILD_DIR}-install
        -DOPENSSL_ROOT_DIR=${CMAKE_PREFIX_PATH}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
        -DCMAKE_INSTALL_MESSAGE:STRING=NEVER
        -DTRITON_KEEP_TYPEINFO:BOOL=ON
        ${client_impl_cmake_extra_args}
    INSTALL_COMMAND ""
    BUILD_ALWAYS OFF
    # BUILD_COMMAND cd ${TRITON_CC_CLIENT_BUILD_DIR} &&
    #     sed -in "s/TRITON_KEEP_TYPEINFO:BOOL=OFF/TRITON_KEEP_TYPEINFO:BOOL=ON/" CMakeCache.txt &&
    #     ${CMAKE_COMMAND} . && cd - && ${CMAKE_COMMAND} --build .
    BUILD_COMMAND bash ${SIRIUS_ROOT_DIR}/client/third_party/triton_cc_client_build_helper.sh ${TRITON_CC_CLIENT_BUILD_DIR}
        # ${CMAKE_COMMAND} --build .

    LOG_DOWNLOAD ON
    LOG_INSTALL ON
    LOG_BUILD ON
)

set(TRITON_3RD_DIR ${TRITON_BUILD_DIR}/third-party)

message(">> TRITON_SOURCE_DIR: ${TRITON_SOURCE_DIR}")
message(">> TRITON_PREFIX: ${TRITON_PREFIX}")
message(">> TRITON_BUILD_DIR: ${TRITON_BUILD_DIR}")
# message(">> TRITON_INSTALL_DIR: ${TRITON_INSTALL_DIR}")
message(">> TRITON_BRANCH_NAME: ${TRITON_BRANCH_NAME}")
message(">> TRITON_3RD_DIR: ${TRITON_3RD_DIR}")

include(ExternalProject)
ExternalProject_Add(client-impl
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/src
    PREFIX ${SIRIUS_BUILD_DIR}/client/impl
    CMAKE_ARGS 
        -DSIRIUS_ROOT_DIR=${SIRIUS_ROOT_DIR}
        -DSIRIUS_BUILD_DIR=${SIRIUS_BUILD_DIR}
        -DTRITON_SOURCE_DIR=${TRITON_SOURCE_DIR}
        -DTRITON_BUILD_DIR=${TRITON_BUILD_DIR}
        -DTRITON_3RD_DIR=${TRITON_3RD_DIR}
        -DTRITON_INSTALL_DIR=${TRITON_BUILD_DIR}-install
        # -DOPENSSL_ROOT_DIR=${CMAKE_PREFIX_PATH}
        # -DCMAKE_INSTALL_MESSAGE=LAZY
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
        -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
        ${client_impl_cmake_extra_args}
        # INSTALL_COMMAND ""
    USES_TERMINAL_BUILD true
    BUILD_ALWAYS ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
    DEPENDS triton-client
    LOG_DOWNLOAD ON
    LOG_INSTALL ON
    LOG_BUILD ON
)

ExternalProject_Add_Step(client-impl build_triton_client 
    COMMAND ${CMAKE_COMMAND} --build ${TRITON_BUILD_DIR}
    DEPENDEES download
    DEPENDERS configure
)
