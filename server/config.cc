#include "config.h"

namespace colserve {

ServeMode Config::serve_mode = ServeMode::kNormal;

ColocateConfig Config::colocate_config = {
  .skip_malloc = false,
  .skip_loading = false,
};

bool Config::check_mps = true;

std::atomic<bool> Config::running = true;

std::string Config::train_profile = "train-profile";

bool Config::use_shared_tensor = true;
bool Config::use_shared_tensor_infer = true;
bool Config::use_shared_tensor_train = true;
bool Config::ondemand_adjust = true;

double Config::cuda_memory_pool_gb = 12;

std::string Config::mempool_freelist_policy= "best-fit";

bool Config::infer_raw_blob_alloc = false;

bool Config::capture_train_log = true;

std::string Config::profile_log_path = "server-profile";

std::string Config::infer_model_config_path = "config";

double Config::train_memory_over_predict_mb = 0;

int Config::train_mps_thread_percent = -1;

size_t Config::max_cache_nbytes = 0 * 1024 * 1024 * 1024;

double Config::memory_pressure_mb = 0;

bool Config::pipeline_load = true;

bool Config::system_initialized = false;

}