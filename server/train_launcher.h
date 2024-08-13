#ifndef COLSERVE_TRAIN_LAUNCHER_H
#define COLSERVE_TRAIN_LAUNCHER_H

#include <server/grpc/grpc_server.h>
#include <server/schedule/job_queue.h>

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
    if (!batch_start_) { batch_start_ = true; }
  }
  void SetTargetBatchSize(int bs) {
    if (batch_start_) target_batch_size_.store(bs, std::memory_order_relaxed);
  }
  int GetCurBatchSize() { 
    return cur_batch_size_; 
  }
  int GetTargetBatchSize() { 
    return target_batch_size_.load(std::memory_order_relaxed); 
  }
  void AddTargetBatchSize(int delta_bs) { 
    if (batch_start_)  target_batch_size_.fetch_add(delta_bs, std::memory_order_relaxed);
  }


  double PredictMemUsageMB(bool verbose);
  int PredictTargetBatchSize(double memory_mb);
  int GetAdjustBatchSize(double memory_mb);

 private:
  static std::unique_ptr<TrainLauncher> train_launcher_;
  std::pair<double, double> GetModelMemParam();

  bool Train();
  bool LaunchTrain(std::shared_ptr<Job> job, std::vector<std::string> &args);
  void DummyAdjust();

  void KillTrain();
  
  JobQueue job_queue_;
  std::unique_ptr<std::thread> thread_;

  std::unique_ptr<std::thread> dummy_adjust_thread_;

  pid_t train_pid_{-1};
  int job_batch_size_{-1};
  int cur_batch_size_{-1};
  std::atomic<int> target_batch_size_{-1};
  bool batch_start_{false};
  std::string cur_model_name_;

  // model -> train code path
  std::unordered_map<std::string, std::filesystem::path> train_handles_;
};

} // namespace colserve

#endif