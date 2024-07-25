# client
add_executable(client client.cc)
target_link_libraries(client grpc++ proto glog::glog CLI11::CLI11)

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
