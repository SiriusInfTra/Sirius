#include <cstdlib>
#include <string>

#include <glog/logging.h>
#include "config.h"


namespace torch_col {

int colocate_use_xsched = 1;
  
int kill_batch_on_recv = 1 & colocate_use_xsched;

int has_colocated_infer_server = 0;

int has_shared_tensor_server = 0;

double shared_tensor_pool_gb = 12;

bool dynamic_sm_partition = false;

void ConfigTorchCol(int use_shared_tensor) {
  static bool configured = false;
  if (configured) {
    return;
  }

  auto has_infer_server_env = std::getenv("HAS_INFER_SERVER");
  auto has_shared_tensor_server_env = std::getenv("HAS_SHARED_TENSOR_SERVER");
  auto pool_size_env = std::getenv("SHARED_TENSOR_POOL_GB");
  auto dynamic_sm_partition_env = std::getenv("DYNAMIC_SM_PARTITION");
  
  has_colocated_infer_server = has_infer_server_env == nullptr ? 
                               false : (std::string(has_infer_server_env) == "1");
  has_shared_tensor_server = has_shared_tensor_server_env == nullptr ? 
                             false : (std::string(has_shared_tensor_server_env) == "1");
  dynamic_sm_partition = dynamic_sm_partition_env == nullptr ? 
                         false : (std::string(dynamic_sm_partition_env) == "1");
  dynamic_sm_partition = dynamic_sm_partition && colocate_use_xsched;
  if (!has_shared_tensor_server && !pool_size_env) {
    LOG(INFO) << "SHARED_TENSOR_POOL_GB not set, use default 12GB";
  } else if (pool_size_env) {
    shared_tensor_pool_gb = std::stod(pool_size_env);
  }

  LOG(INFO) << "TorchColConfig::use_shared_tensor: " << use_shared_tensor;
  LOG(INFO) << "TorchColConfig::colocate_use_xsched: " << colocate_use_xsched;
  LOG(INFO) << "TorchColConfig::has_colocated_infer_server:" << has_colocated_infer_server;
  LOG(INFO) << "TorchColConfig::has_shared_tensor_server:" << has_shared_tensor_server;
  LOG(INFO) << "TorchColConfig::shared_tensor_pool_gb:" << shared_tensor_pool_gb;
  LOG(INFO) << "TorchColConfig::dynamic_sm_partition:" << dynamic_sm_partition;
  configured = true;
}

}