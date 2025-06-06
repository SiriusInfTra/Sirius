#include <filesystem>
#include <glog/logging.h>

#include "util.h"

namespace colserve {
namespace workload {

std::filesystem::path data_repo_path{"client/data"};

std::string ReadInput(std::filesystem::path data_path) {
  data_path = data_repo_path / data_path;
  std::ifstream data_file{data_path, std::ios::binary};
  CHECK(data_file.good()) << "data " << data_path << " not exist";
  std::string data{std::istreambuf_iterator<char>(data_file), std::istreambuf_iterator<char>()};
  data_file.close();
  return data;
}

int GetLLMMaxModelLen(const std::string &model_name) {
  if (model_name == "meta-llama/Meta-Llama-3-8B") {
    return 8192;
  } else if (model_name == "meta-llama/Meta-Llama-3-8B-Instruct") {
    return 8192;
  } else if (model_name == "meta-llama/Llama-2-7b-hf" 
      || model_name == "meta-llama/Llama-2-13b-hf") {
    return 4096;
  } else if (model_name == "facebook/opt-125m") {
    return 2048;
  }
  LOG(FATAL) << "Unknown model name: " << model_name;
} 

std::string GetLLMPromptJsonPath(const std::string &model) {
  if (model == "meta-llama/Meta-Llama-3-8B"
      || model == "meta-llama/Meta-Llama-3-8B-Instruct"
      || model == "facebook/opt-125m") {
    return "client/data/sharegpt/prompt_with_length_llama3.json";
  } else if (model == "meta-llama/Llama-2-7b-hf" 
      || model == "meta-llama/Llama-2-13b-hf") {
    return "client/data/sharegpt/prompt_with_length_llama2.json";
  }
  LOG(FATAL) << "Unknown model name: " << model;
}

bool IsLLM(const std::string &model_name) {
  return model_name.find("llama") != std::string::npos
    || model_name.find("opt") != std::string::npos;
}

AppBase::AppBase(const std::string &name) : app{name} {
  app.add_flag("--infer,!--no-infer", enable_infer, "enable infer workload");
  app.add_flag("--train,!--no-train", enable_train, "enable train workload");
  app.add_option("--train-model", train_models, "models of train workload");
  app.add_option("-d,--duration", duration, "duration of workload");
  app.add_option("-c,--concurrency", concurrency, "concurrency of infer workload");
  app.add_option("--num-epoch", num_epoch, "num_epoch of train workload");
  app.add_option("--batch-size", batch_size, "batch_size of train workload");
  app.add_option_no_stream("-l,--log", log, "log file");
  app.add_option("-v,--verbose", verbose, "verbose level");
  app.add_option("--show-result", show_result, "show result");
  app.add_option("--ip", sirius_ip, "sirius grpc ip");
  app.add_option( "-p,--port", sirius_port, "sirius grpc port");
  app.add_option("--triton-ip", triton_ip, "triton grpc ip");
  app.add_option("--triton-port", triton_port, "triton grpc port");
  app.add_option("--triton-max-memory", triton_max_memory, "max memory of triton server(MB)");
  app.add_option("--triton-config", triton_config, "triton config file");
  app.add_option("--triton-device-map", triton_device_map, "triton device map file");
  
  app.add_option("--seed", seed, "random seed");

  app.add_option("--warmup", warmup, "warm up infer model");
  app.add_option("--wait-warmup-done-sec", wait_warmup_done_sec, "trace file of infer workload");

  app.add_option("--wait-train-setup-sec", wait_train_setup_sec, "delay before start infer.");
  app.add_option("--wait-stable-before-start-profiling-sec", wait_stable_before_start_profiling_sec, "trace file of infer workload");

  app.add_option("--infer-timeline", infer_timeline, "path of infer timeline");
}

uint64_t AppBase::seed = static_cast<uint64_t>(-1);

}  // namespace workload
}
