#include "config.h"

namespace colserve {

ServeMode Config::serve_mode = ServeMode::kNormal;

ColocateConfig Config::colocate_config = {
  .skip_malloc = false,
  .skip_loading = false,
};

bool Config::check_mps = true;

std::atomic<bool> Config::running = true;

bool Config::use_shared_tensor = true;
bool Config::use_shared_tensor_infer = true;
bool Config::use_shared_tensor_train = true;

double Config::cuda_memory_pool_gb = 12;

bool Config::infer_raw_blob_alloc = true;

bool Config::capture_train_log = true;

std::string Config::profile_log_path = "server-profile";

std::string Config::infer_model_config_path = "config";

int Config::train_mps_thread_percent = -1;

}