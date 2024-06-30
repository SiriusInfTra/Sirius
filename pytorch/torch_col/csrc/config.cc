#include <cstdlib>
#include <string>

#include <glog/logging.h>
#include "config.h"


namespace torch_col {

int TorchColConfig::use_shared_tensor = false;
double TorchColConfig::shared_tensor_pool_gb = 12;

int TorchColConfig::has_colocated_infer_server = 0;
int TorchColConfig::has_shared_tensor_server = 0;

int TorchColConfig::colocate_use_xsched = 0;
int TorchColConfig::kill_batch_on_recv = 0;
  
int TorchColConfig::dynamic_sm_partition = false;

std::string TorchColConfig::hook_mode = "none";


void TorchColConfig::InitConfig(int use_shared_tensor_) {
  static bool configured = false;
  if (configured) {
    return;
  }

  use_shared_tensor = use_shared_tensor;
  auto colocate_use_xsched_env = std::getenv("COL_USE_XSCHED");
  auto has_infer_server_env = std::getenv("COL_HAS_INFER_SERVER");
  auto has_shared_tensor_server_env = std::getenv("COL_HAS_SHARED_TENSOR_SERVER");
  auto pool_size_env = std::getenv("COL_SHARED_TENSOR_POOL_GB");
  auto dynamic_sm_partition_env = std::getenv("COL_DYNAMIC_SM_PARTITION");
  auto hook_mode_env = std::getenv("COL_HOOK_MODE");
  
  colocate_use_xsched = colocate_use_xsched_env == nullptr ? 
                        false : (std::string(colocate_use_xsched_env) == "1");
  has_colocated_infer_server = has_infer_server_env == nullptr ? 
                               false : (std::string(has_infer_server_env) == "1");
  has_shared_tensor_server = has_shared_tensor_server_env == nullptr ? 
                             false : (std::string(has_shared_tensor_server_env) == "1");
  dynamic_sm_partition = dynamic_sm_partition_env == nullptr ? 
                         false : (std::string(dynamic_sm_partition_env) == "1");
  dynamic_sm_partition = dynamic_sm_partition && colocate_use_xsched;
  hook_mode = hook_mode_env == nullptr ? "none" : std::string(hook_mode_env);

  if (hook_mode == "xsched-sync2") {
    kill_batch_on_recv = 1 && colocate_use_xsched;
  } else {
    kill_batch_on_recv = 0;
  }

  if (!has_shared_tensor_server && !pool_size_env) {
    LOG(INFO) << "COL_SHARED_TENSOR_POOL_GB not set, use default 12GB";
  } else if (pool_size_env) {
    shared_tensor_pool_gb = std::stod(pool_size_env);
  }

  LOG(INFO) << "TorchColConfig::use_shared_tensor=" << use_shared_tensor;
  LOG(INFO) << "TorchColConfig::colocate_use_xsched=" << colocate_use_xsched;
  LOG(INFO) << "TorchColConfig::kill_batch_on_recv=" << kill_batch_on_recv;
  LOG(INFO) << "TorchColConfig::has_colocated_infer_server=" << has_colocated_infer_server;
  LOG(INFO) << "TorchColConfig::has_shared_tensor_server=" << has_shared_tensor_server;
  LOG(INFO) << "TorchColConfig::shared_tensor_pool_gb=" << shared_tensor_pool_gb;
  LOG(INFO) << "TorchColConfig::dynamic_sm_partition=" << dynamic_sm_partition;
  LOG(INFO) << "TorchColConfig::hook_mode=" << hook_mode;
  configured = true;
}

}