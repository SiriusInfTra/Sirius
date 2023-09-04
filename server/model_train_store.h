#ifndef COLSERVE_MODEL_TRAIN_STORE_H
#define COLSERVE_MODEL_TRAIN_STORE_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <future>

#include "grpc/grcp_server.h"
#include "job_queue.h"

namespace colserve {

class ModelTrainStore {
 public:
  static void Init(const std::filesystem::path &train_store_path);
  static ModelTrainStore* Get() { 
    if (model_train_store_ == nullptr) {
      LOG(FATAL) << "ModelTrainStore not initialized";
    }
    return model_train_store_.get();
  }
  static bool Shutdown();

  bool AddJob(network::TrainHandler::TrainData* data);
  pid_t GetTrainPid() { return train_pid_; }

 private:
  static std::unique_ptr<ModelTrainStore> model_train_store_;

  bool Train();
  bool LaunchTrain(std::shared_ptr<Job> job, std::vector<std::string> &args);
  
  JobQueue job_queue_;
  std::unique_ptr<std::thread> thread_;
  pid_t train_pid_{-1};

  // model -> train code path
  std::unordered_map<std::string, std::filesystem::path> train_handles_;
};

} // namespace colserve

#endif