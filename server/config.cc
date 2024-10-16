#include <common/util.h>

#include <server/config.h>

namespace colserve {

ServeMode Config::serve_mode = ServeMode::kNormal;
ModelPlacePolicy Config::model_place_policy = ModelPlacePolicy::kRoundRobin;

std::filesystem::path Config::binary_directory;

ColocateConfig Config::colocate_config = {
  .skip_malloc = false,
  .skip_loading = false,
};

bool Config::check_mps = true;

std::atomic<bool> Config::running{true};

std::string Config::train_profile = "train-profile.csv";

bool Config::use_xsched = false;
std::string colserve::Config::port = "8080";
bool Config::use_shared_tensor = true;
bool Config::no_infer = false;
bool Config::use_shared_tensor_infer = true;
bool Config::use_shared_tensor_train = true;
bool Config::ondemand_adjust = true;
bool Config::better_alloc = true;

// size_t Config::better_alloc_threshold = 64_MB;

bool Config::group_param_load = true;
bool Config::group_param_dump = false;
bool Config::group_param_nbytes_with_fragment = true;
size_t Config::group_param_load_threshold = 8_MB;

double Config::cuda_memory_pool_gb = 12;

std::string Config::mempool_freelist_policy= "best-fit";

bool Config::infer_raw_blob_alloc = false;

bool Config::capture_train_log = true;

std::string Config::profile_log_path = "server-profile";

std::string Config::infer_model_config_path = "config";

double Config::train_memory_over_predict_mb = 0;

int Config::train_mps_thread_percent = -1;

size_t Config::cold_cache_min_capability_nbytes = 0_GB;
size_t Config::cold_cache_max_capability_nbytes = 0_GB;

double Config::cold_cache_ratio = 0.3;
size_t Config::train_over_adjust_nbytes = 500_MB;

size_t Config::max_warm_cache_nbytes = 0_GB;

bool Config::enable_warm_cache_fallback = true;

double Config::memory_pressure_mb = 0;

bool Config::pipeline_load = true;

double Config::task_switch_delay_ms = 10;

bool Config::has_warmup = false;
double Config::infer_model_max_idle_ms = 3000;

bool Config::dump_adjust_info = false;

bool Config::profiler_acquire_resource_lock = false;

bool Config::skip_set_mps_thread_percent = false;
bool Config::dynamic_sm_partition = false;
bool Config::estimate_infer_model_tpc = false;
double Config::infer_exec_time_estimate_scale = 1.1;

bool Config::enable_train_adjust_balance = true;
memory_mb_t Config::train_adjust_balance_threshold = 1500;

bool Config::dummy_adjust = false;

bool Config::system_initialized = false;

bool Config::profile_gpu_smact = true;
bool Config::profile_gpu_util = true;
bool Config::profile_sm_partition = false;

bool Config::log_all = false;
bool Config::log_grpc = false;
bool Config::log_infer_sched = false;
bool Config::log_train_init = true;
bool Config::log_warm_cache = false;
bool Config::log_cold_cache = false;
bool Config::log_infer_model_init = false;
bool Config::log_infer_model_reclaim = false;
bool Config::log_infer_time = false;
bool Config::log_infer_pipeline_exec = false;
bool Config::log_infer_load_param = false;
bool Config::log_memory_adjust = false;
bool Config::log_controller = false;
bool Config::log_task_switch = false;

}