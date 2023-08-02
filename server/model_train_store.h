#ifndef COLSERVE_MODEL_TRAIN_STORE_H
#define COLSERVE_MODEL_TRAIN_STORE_H

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <thread>

#include "grpc/grcp_server.h"
#include "job_queue.h"

namespace colserve {

class ModelTrainStore {
 public:
  static void Init(const std::filesystem::path &train_store_path);
  static ModelTrainStore* Get() { return model_train_store_.get();}

  bool AddJob(network::TrainHandler::TrainData* data);
  
 private:
  static std::unique_ptr<ModelTrainStore> model_train_store_;

  bool Train();
  
  JobQueue job_queue_;
  std::unique_ptr<std::thread> thread_;

  // model -> train code path
  std::unordered_map<std::string, std::filesystem::path> train_handles_;
};

} // namespace colserve

#endif