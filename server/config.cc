#include "config.h"

namespace colserve {

ServeMode Config::serve_mode = ServeMode::kNormal;

bool Config::check_mps = true;

std::atomic<bool> Config::running = true;

bool Config::use_shared_tensor = true;
bool Config::use_shared_tensor_infer = true;
bool Config::use_shared_tensor_train = true;

double Config::cuda_memory_pool_gb = 12;

bool Config::infer_raw_blob_alloc = true;

std::string Config::profile_log_path = "server-profile";

std::string Config::infer_model_config_path = "config";

}