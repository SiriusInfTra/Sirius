#ifndef COLSERVE_TRAIN_LAUNCHER_H
#define COLSERVE_TRAIN_LAUNCHER_H

#include <server/grpc/grpc_server.h>
#include <server/job_queue.h>

#include <iostream>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <future>
#include <atomic>

namespace colserve {

class TrainLauncher {
 public:
  static void Init(const std::filesystem::path &train_store_path);
  static TrainLauncher* Get() { 
    if (train_launcher_ == nullptr) {
      LOG(FATAL) << "TrainLauncher not initialized";
    }
    return train_launcher_.get();
  }
  static bool Shutdown();

  bool AddJob(network::TrainHandler::TrainData* data);
  pid_t GetTrainPid() { return train_pid_; }

  void SetCurBatchSize(int bs) {
    cur_batch_size_ = bs; 
    if (first_batch_) {target_batch_size_ = bs; first_batch_ = false; }
  }
  int GetCurBatchSize() { return cur_batch_size_; }
  int GetTargetBatchSize() { return target_batch_size_.load(std::memory_order_relaxed); }
  void AddTargetBatchSize(int delta_bs) { 
    if (!first_batch_)  target_batch_size_.fetch_add(delta_bs, std::memory_order_relaxed);
  }

  double PredictMemUsageMB();

 private:
  static std::unique_ptr<TrainLauncher> train_launcher_;

  bool Train();
  bool LaunchTrain(std::shared_ptr<Job> job, std::vector<std::string> &args);
  void DummyAdjust();
  
  JobQueue job_queue_;
  std::unique_ptr<std::thread> thread_;

  std::unique_ptr<std::thread> dummy_adjust_thread_;

  pid_t train_pid_{-1};
  int cur_batch_size_{-1};
  std::atomic<int> target_batch_size_{-1};
  bool first_batch_{true};
  std::string cur_model_name_;

  // model -> train code path
  std::unordered_map<std::string, std::filesystem::path> train_handles_;
};

} // namespace colserve

#endif