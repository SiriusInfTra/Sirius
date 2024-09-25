# client
add_executable(client client.cc)
target_link_libraries(client grpc++ proto glog::glog CLI11::CLI11)
include(ExternalProject)
add_executable(workload_launcher 
  workload_launcher.cc
  workload/workload.cc
  workload/util.cc
)
target_link_libraries(workload_launcher grpc++ proto glog::glog CLI11::CLI11)

add_executable(busy_loop_launcher 
  busy_loop_launcher.cc
  workload/workload.cc
  workload/util.cc
)
target_link_libraries(busy_loop_launcher grpc++ proto glog::glog CLI11::CLI11)

set(TRITON_COMMON_REPO_TAG "r23.12")
set(TRITON_THIRD_PARTY_REPO_TAG "r23.12")
set(TRITON_CORE_REPO_TAG "r23.12")
set(TRITON_USE_THIRD_PARTY ON)
ExternalProject_Add(triton-client
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/triton-client
    CMAKE_ARGS 
        # -DCMAKE_INSTALL_PREFIX=${CMAKE}/install
        -DTRITON_ENABLE_CC_HTTP=OFF 
        -DTRITON_ENABLE_CC_GRPC=ON 
        -DTRITON_ENABLE_PERF_ANALYZER=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_C_API=OFF 
        -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF
        -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF
        -DTRITON_ENABLE_GPU=ON 
        -DTRITON_ENABLE_EXAMPLES=OFF 
        -DTRITON_ENABLE_TESTS=OFF
        -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG}
        -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG}
        -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG}
        -DTRITON_USE_THIRD_PARTY=${TRITON_USE_THIRD_PARTY}
    # 如果需要传递其他 CMake 选项，可以在这里添加
    INSTALL_COMMAND ""
)

