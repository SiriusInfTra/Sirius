#ifndef COLSERVE_JOB_QUEUE_H
#define COLSERVE_JOB_QUEUE_H

#include <server/logging_as_glog.h>
#include <server/grpc/grpc_server.h>
#include <server/profiler.h>

#include <iostream>
#include <queue>
#include <memory>
#include <vector>
#include <shared_mutex>
#include <mutex>
#include <condition_variable>
#include <chrono>


namespace colserve {

class Job {
 public:
  virtual network::InferHandler::InferData* GetInferData() { 
    LOG(FATAL) << "not implemented"; return nullptr; 
  }
  virtual network::TrainHandler::TrainData* GetTrainData() { 
    LOG(FATAL) << "not implemented"; return nullptr; 
  }
  virtual std::ostream& Print(std::ostream& os) const { LOG(FATAL) << "not implemented"; }
  inline void RecordEnqueued() { en_queue_time_ = std::chrono::steady_clock::now(); }
  inline void RecordDequeued() { de_queue_time_ = std::chrono::steady_clock::now(); }
  inline void RecordFinished() { finish_time_ = std::chrono::steady_clock::now(); }
  inline virtual void RecordProfile() {
    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferJobQueue, 
        std::chrono::duration<double, std::milli>(de_queue_time_ - en_queue_time_).count());
    Profiler::Get()->RecordPerf(Profiler::PerfItem::InferJobProcess,
        std::chrono::duration<double, std::milli>(finish_time_ - de_queue_time_).count());
  }
  std::chrono::time_point<std::chrono::steady_clock> GetEnQueueTime() {
    return en_queue_time_;
  }
 protected:
  std::chrono::time_point<std::chrono::steady_clock> en_queue_time_, 
                                                     de_queue_time_,
                                                     finish_time_; 

};

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Job>& job);

class InferJob : public Job {
 public:
  InferJob(network::InferHandler::InferData* data);
  network::InferHandler::InferData* GetInferData() override { return data_; }

 protected:
  network::InferHandler::InferData* data_;
  virtual std::ostream& Print(std::ostream& os) const;
};

struct LLMRequestMetric;
class LLMInferJob : public InferJob {
 public:
  LLMInferJob(network::InferHandler::InferData* data);
  std::string_view GetPrompt() {
    return prompt_;
  }
  int GetMaxTokens() {
    return max_tokens_;
  }
  void RecordProfile(const LLMRequestMetric &metric);

 private:
  std::string_view prompt_;
  int max_tokens_;
};

class TrainJob : public Job {
 public:
  TrainJob(network::TrainHandler::TrainData* data);
  network::TrainHandler::TrainData* GetTrainData() override { return data_; }

 private:
  network::TrainHandler::TrainData* data_;
  virtual std::ostream& Print(std::ostream& os) const;
};

class JobQueue {
 public:
  bool Put(const std::shared_ptr<Job> &job);
  std::shared_ptr<Job> Get();
  size_t NumJobs();
  
 protected:
  std::mutex mutex_;
  std::queue<std::shared_ptr<Job>> queue_;
};

class BatchJobQueue : public JobQueue {
 public:
  std::vector<std::shared_ptr<Job>> GetBatch(
      size_t batch_size, size_t interval_ms = 0, 
      size_t timeout_ms = std::numeric_limits<int64_t>::max());
  bool Put(const std::shared_ptr<Job> &job);
  double FirstJobQueueTime(); // ms
  
 private:
  std::condition_variable put_job_cv_;
};

}

#endif