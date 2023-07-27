#ifndef COLSERVE_JOB_QUEUE_H
#define COLSERVE_JOB_QUEUE_H

#include <iostream>
#include <queue>
#include <memory>
#include <vector>
#include <shared_mutex>
#include <mutex>
#include <condition_variable>

#include "grpc/grcp_server.h"

namespace colserve {

class Job {
 public:
  virtual network::InferHandler::InferData* GetInferData() { LOG(FATAL) << "not implemented"; }
  virtual network::TrainHandler::TrainData* GetTrainData() { LOG(FATAL) << "not implemented"; }
  virtual std::ostream& Print(std::ostream& os) const { LOG(FATAL) << "not implemented"; }
 protected:
};

class InferJob : public Job {
 public:
  InferJob(network::InferHandler::InferData* data);
  network::InferHandler::InferData* GetInferData() override { return data_; }

 private:
  network::InferHandler::InferData* data_;
  virtual std::ostream& Print(std::ostream& os) const;
};

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Job>& job);

class JobQueue {
 public:
  bool Put(const std::shared_ptr<Job> &job);
  std::shared_ptr<Job> Get();
  
 protected:
  std::mutex mutex_;
  std::queue<std::shared_ptr<Job>> queue_;
};

class BatchJobQueue : public JobQueue {
 public:
  std::vector<std::shared_ptr<Job>> GetBatch(size_t batch_size, size_t timeout_ms = 0);
  bool Put(const std::shared_ptr<Job> &job);

 private:
  std::condition_variable put_job_;
  std::queue<std::shared_ptr<Job>> queue_;
};

}

#endif