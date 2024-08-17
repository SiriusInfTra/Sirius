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

std::string GetModelNameWithoutDuplicatedId(const std::string &model_name);

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
  static int GetPreEstimatedTPC(const std::string &model_name);

  Model() : name_("dummy") {};
  // Model(const std::string &name, const std::filesystem::path &model_path, DLDevice device, 
  //       size_t batch_size, size_t num_worker, size_t max_num_worker);
  Model(const std::string &name, const std::filesystem::path &model_path,
        std::optional<const std::map<std::string, tvm::TVMArray>> params,
        DLDevice device, size_t batch_size, 
        size_t num_worker, size_t max_num_worker);

  size_t NumJobs() { return job_queue_.NumJobs(); }

  bool ReclaimMemory(size_t rank, std::unique_lock<std::mutex> &cold_cache_lock, 
                  std::unique_lock<std::mutex> &model_lock, Model *source_model);

  bool ReclaimMemory(size_t rank, Model *source_model);

  void ClearColdCache(const std::vector<size_t> &cold_cached_group_id, int rank,
                                  std::unique_lock<std::mutex> &cold_cache_lock);

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

  friend class InferModelStore;
  friend class WarmModelCache;
  
 private:

  bool AddJob(network::InferHandler::InferData* data);

  void InitMetaInfo();
  bool MaybeAdjustTrainAndCache(size_t rank, 
                                std::unique_lock<std::mutex> 
                                &cold_cache_lock, 
                                std::unique_lock<std::mutex> &model_lock);
  bool SetupMemory(size_t rank, 
                   std::unique_lock<std::mutex> &cold_cache_lock, 
                   std::unique_lock<std::mutex> &model_lock);
  bool Inference(uint32_t rank, pthread_barrier_t* barrier);
  bool SetInput(tvm::Executor &graph_executor, size_t idx, const std::string &input_id, 
                const std::vector<std::shared_ptr<Job>> &jobs);
  bool GetOutput(tvm::Executor &graph_executor, 
                 size_t idx, const std::string &output_id, 
                 const std::vector<std::shared_ptr<Job>> &jobs);

  void ChangeStatus(uint32_t rank, Status to) {
    auto from = status_[rank];
    status_[rank] = to;
    model_stat_[static_cast<size_t>(from)].fetch_sub(1, std::memory_order_relaxed);
    model_stat_[static_cast<size_t>(to)].fetch_add(1, std::memory_order_relaxed);
  }

  void EstimateTPC(uint32_t rank, tvm::Executor &graph_executor);
  void WaitEstimateTPC();  

  // guarentee only estimate one model at a time
  static std::mutex estimate_tpc_mut_;
  static std::condition_variable estimate_tpc_cv_;
  static bool estimating_tpc_;
  
  constexpr static int MAX_NUM_WORKER = 8;
  static std::array<std::atomic<int>, 
                    static_cast<size_t>(Status::kNumStatus)> model_stat_; 

  std::string name_;
  DLDevice device_;
  size_t batch_size_;
  BatchJobQueue job_queue_;

  // bool warmup_;
  uint32_t max_num_worker_;
  std::atomic<uint32_t> num_worker_{0};

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
  
  int required_num_tpc_{-1};
};

}

#endif