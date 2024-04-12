#ifndef COLSERVE_INFER_MODEL_H
#define COLSERVE_INFER_MODEL_H

#include <server/tvm/executor.h>
#include <server/grpc/grpc_server.h>

#include <mutex>
#include <string>
#include <optional>
#include <filesystem>
#include <vector>
#include <thread>
#include <unordered_map>

namespace colserve {

class Model {
 public:
  enum class Status {
    kWithoutMemory,
    kWithoutParam,
    kReady,
    kNumStatus
  };

  static int GetNumModel(Status status) {
    return model_stat_[static_cast<size_t>(status)].load(std::memory_order_relaxed);
  }

  Model() : name_("dummy") {};
  // Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device, 
  //       size_t batch_size, size_t num_worker, size_t max_num_worker);
  Model(const std::string &name, const std::filesystem::path &model_path,
        std::optional<const std::map<std::string, tvm::TVMArray>> params,
        DLDevice device, size_t batch_size, size_t num_worker, size_t max_num_worker);

  size_t NumJobs() { return job_queue_.NumJobs(); }

  bool ReclaimMemory(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, std::unique_lock<std::mutex> &model_lock);

  void ClearColdCache(const std::vector<size_t> &cold_cached_group_id, int rank, std::unique_lock<std::mutex> &cold_cache_lock);

  const std::string &GetName() { return name_; }
  double GetIdleMill(size_t rank) {
    return infer_idle_mills_[rank].load(std::memory_order_relaxed);
  }
  size_t GetHotness() {
    return infer_count_.load(std::memory_order_relaxed);
  }

  size_t GetMemoryNbytes(size_t rank) {
    return executors_[rank]->GetStorageSizeAlign();
  }

  // void SetWaitTrainPid(size_t worker_id, pid_t train_pid) {
  //   CHECK_LT(worker_id, waited_trains_.size());
  //   waited_trains_[worker_id] = train_pid;
  // }

  friend class InferModelStore;
  friend class WarmModelCache;
  
 private:
  bool AddJob(network::InferHandler::InferData* data);

  void InitMetaInfo();
  bool MaybeAdjustTrainAndCache(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, std::unique_lock<std::mutex> &model_lock);
  bool SetupMemory(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, std::unique_lock<std::mutex> &model_lock);
  bool Inference(uint32_t rank, pthread_barrier_t* barrier);
  bool SetInput(tvm::Executor &graph_executor, size_t idx, const std::string &input_id, 
                const std::vector<std::shared_ptr<Job>> &jobs);
  bool GetOutput(tvm::Executor &graph_executor, 
                 size_t idx, const std::string &output_id, const std::vector<std::shared_ptr<Job>> &jobs);

  void ChangeStatus(uint32_t rank, Status to) {
    auto from = status_[rank];
    status_[rank] = to;
    model_stat_[static_cast<size_t>(from)].fetch_sub(1, std::memory_order_relaxed);
    model_stat_[static_cast<size_t>(to)].fetch_add(1, std::memory_order_relaxed);
  }
  // inline double GetMaxIdleMill() { 
  //   if (warmup_) {
  //     return 3000; // a default dummy value
  //   }
  //   return scale_down_idle_time_; 
  // }
  // void MonitorJob();
  
  constexpr static int MAX_NUM_WORKER = 8;
  static std::array<std::atomic<int>, 
                    static_cast<size_t>(Status::kNumStatus)> model_stat_; 

  std::string name_;
  DLDevice device_;
  size_t batch_size_;
  BatchJobQueue job_queue_;

  // bool warmup_;
  uint32_t max_num_worker_;
  std::atomic<uint32_t> num_worker_;

  std::unique_ptr<tvm::TVMGraph> tvm_graph_ = nullptr;
  // param_name -> [[shape], dtype]
  std::unordered_map<std::string, 
      std::pair<std::vector<int64_t>, std::string>> input_info_, output_info_;

  // need to lock when changing model resources
  std::array<std::mutex, MAX_NUM_WORKER> muts_;
  std::vector<Status> status_;
  std::vector<std::unique_ptr<tvm::Executor>> executors_;
  std::vector<std::unique_ptr<std::thread>> infer_workers_;
  std::array<std::atomic<double>, MAX_NUM_WORKER> infer_idle_mills_;

  std::atomic<size_t> infer_count_{0};


  // std::vector<std::unique_ptr<tvm::Executor>> graph_executor_pool_;
  // std::unique_ptr<tvm::TVMGraph> graph_executor_factory_;




  // infer scaling
  // double scale_up_queue_time_;  // ms
  // double scale_down_idle_time_; // ms

  // std::vector<std::unique_ptr<std::atomic<bool>>> worker_running_;

  // std::vector<pid_t> waited_trains_;


  // std::unique_ptr<std::thread> thread_;
  // std::unique_ptr<std::thread> job_monitor_;
  
};

}

#endif