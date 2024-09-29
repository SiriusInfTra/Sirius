#include <torch_col/csrc/config.h>

#include <common/log_as_glog_sta.h>

#include <boost/format.hpp>
#include <cstdlib>
#include <ostream>
#include <iostream>
#include <sstream>
#include <string>


namespace torch_col {

int TorchColConfig::use_shared_tensor = false;
double TorchColConfig::shared_tensor_pool_gb = 12;

int TorchColConfig::has_colocated_infer_server = 0;
int TorchColConfig::has_shared_tensor_server = 0;

int TorchColConfig::colocate_use_xsched = 0;
int TorchColConfig::kill_batch_on_recv = 0;
  
int TorchColConfig::dynamic_sm_partition = false;

std::string TorchColConfig::colocate_ctrl_hook_mode = "none";
std::string TorchColConfig::colocate_train_mode = "normal";

int TorchColConfig::release_interm_memory_by_grad_fn = false;
int TorchColConfig::release_interm_memory_by_tagging = true;
int TorchColConfig::use_fbward_hook = true;

int TorchColConfig::train_rank = 0;
int TorchColConfig::train_world_size = 1;

std::string TorchColConfig::train_profile_log_path = "";

bool TorchColConfig::log_all = false;
bool TorchColConfig::log_dynamic_batch = false;
bool TorchColConfig::log_control_stub = false;

int TorchColConfig::configured = false;

void TorchColConfig::InitConfig(int train_rank_, int train_world_size_) {
  if (TorchColConfig::configured) {
    return;
  }

  TorchColConfig::train_rank = train_rank_;
  TorchColConfig::train_world_size = train_world_size_;

  auto use_shared_tensor_env = std::getenv("COL_USE_SHARED_TENSOR");
  auto colocate_use_xsched_env = std::getenv("COL_USE_XSCHED");
  auto has_infer_server_env = std::getenv("COL_HAS_INFER_SERVER");
  auto has_shared_tensor_server_env = std::getenv("COL_HAS_SHARED_TENSOR_SERVER");
  auto pool_size_env = std::getenv("COL_SHARED_TENSOR_POOL_GB");
  auto dynamic_sm_partition_env = std::getenv("COL_DYNAMIC_SM_PARTITION");
  auto hook_mode_env = std::getenv("COL_HOOK_MODE");
  auto train_mode_env = std::getenv("COL_TRAIN_MODE");
  auto train_profile_log_path_env = std::getenv("COL_TRAIN_PROFILE_LOG_PATH");
  auto log_all_env = std::getenv("COL_LOG_ALL");
  auto log_dynamic_batch_env = std::getenv("COL_LOG_DYNAMIC_BATCH");
  auto log_control_stub_env = std::getenv("COL_LOG_CONTROL_STUB");

  use_shared_tensor = use_shared_tensor_env == nullptr ? 
                      false : (std::string(use_shared_tensor_env) == "1");
  colocate_use_xsched = colocate_use_xsched_env == nullptr ? 
                        false : (std::string(colocate_use_xsched_env) == "1");
  has_colocated_infer_server = has_infer_server_env == nullptr ? 
                               false : (std::string(has_infer_server_env) == "1");
  has_shared_tensor_server = has_shared_tensor_server_env == nullptr ? 
                             false : (std::string(has_shared_tensor_server_env) == "1");
  dynamic_sm_partition = dynamic_sm_partition_env == nullptr ? 
                         false : (std::string(dynamic_sm_partition_env) == "1");
  dynamic_sm_partition = dynamic_sm_partition && colocate_use_xsched;
  colocate_ctrl_hook_mode = hook_mode_env == nullptr ? "none" : std::string(hook_mode_env);
  colocate_train_mode = train_mode_env == nullptr ? "normal" : std::string(train_mode_env);
  train_profile_log_path = train_profile_log_path_env == nullptr ? 
                           "" : std::string(train_profile_log_path_env);

  log_all = log_all_env == nullptr ? 
            false : (std::string(log_all_env) == "1");
  log_dynamic_batch = log_all || (log_dynamic_batch_env == nullptr ?
                      false : (std::string(log_dynamic_batch_env) == "1"));
  log_control_stub = log_all || (log_control_stub_env == nullptr ?
                      false : (std::string(log_control_stub_env) == "1"));

  if (colocate_ctrl_hook_mode == "xsched-sync2") {
    kill_batch_on_recv = 1 && colocate_use_xsched;
  } else {
    kill_batch_on_recv = 0;
  }

  if (!has_shared_tensor_server && !pool_size_env) {
    LOG(INFO) << "COL_SHARED_TENSOR_POOL_GB not set, use default 12GB";
  } else if (pool_size_env) {
    shared_tensor_pool_gb = std::stod(pool_size_env);
  }

  auto config_head = (boost::format(
      "================ TORCH_COL CONFIG [Rank %d PID %d] ================") 
        % train_rank % getpid()).str();

  std::stringstream config_ss;
  config_ss << config_head << std::endl;
  config_ss << "TorchColConfig::rank=" << train_rank 
            << "|world_size=" << train_world_size << std::endl;;
  config_ss << "TorchColConfig::use_shared_tensor=" 
            << use_shared_tensor << std::endl;
  config_ss << "TorchColConfig::colocate_use_xsched=" << colocate_use_xsched 
            << "|LD_LIBRARY_PATh=" 
            << (getenv("LD_LIBRARY_PATH") != nullptr ? getenv("LD_LIBRARY_PATH") : "")
            << std::endl;
  config_ss << "TorchColConfig::kill_batch_on_recv=" 
            << kill_batch_on_recv << std::endl;
  config_ss << "TorchColConfig::has_colocated_infer_server=" 
            << has_colocated_infer_server << std::endl;
  config_ss << "TorchColConfig::has_shared_tensor_server=" 
            << has_shared_tensor_server << std::endl;
  config_ss << "TorchColConfig::shared_tensor_pool_gb=" 
            << shared_tensor_pool_gb << std::endl;
  config_ss << "TorchColConfig::dynamic_sm_partition=" 
            << dynamic_sm_partition << std::endl;
  config_ss << "TorchColConfig::colocate_ctrl_hook_mode=" 
            << colocate_ctrl_hook_mode << std::endl;
  config_ss << "TorchColConfig::colocate_train_mode=" 
            << colocate_train_mode << std::endl;
  config_ss << "TorchColConfig::train_profile_log_path=" 
            << train_profile_log_path << std::endl;
  config_ss << std::string(config_head.size(), '=') << std::endl;
  std::cerr << config_ss.str() << std::endl;

  TorchColConfig::configured = true;
}

}