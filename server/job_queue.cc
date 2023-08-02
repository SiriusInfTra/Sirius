#include <chrono>
#include <glog/logging.h>

#include "grpc/grcp_server.h"
#include "job_queue.h"

namespace colserve {

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Job>& job) {
  job->Print(os);
  return os;
}

InferJob::InferJob(network::InferHandler::InferData* data)
    : data_(data) {
}

std::ostream& InferJob::Print(std::ostream& os) const {
  os << "InferJob [" << data_->GetModelName() << ", " << data_->GetId() << "]";
  return os;
}

TrainJob::TrainJob(network::TrainHandler::TrainData* data) 
    : data_(data) {
}

std::ostream& TrainJob::Print(std::ostream& os) const {
  os << "TrainJob [" << data_->GetModelName() << ", " << data_->GetId() << "]";
  return os;
}

bool JobQueue::Put(const std::shared_ptr<Job> &job) {
  std::unique_lock lock{mutex_};
  queue_.push(job);
  return true;
}

std::shared_ptr<Job> JobQueue::Get() {
  std::unique_lock lock{mutex_};
  if (queue_.empty()) {
    return nullptr;
  } else {
    auto job = queue_.front();
    queue_.pop();
    return job;
  }
}

bool BatchJobQueue::Put(const std::shared_ptr<Job> &job) {
  std::unique_lock lock{mutex_};
  queue_.push(job);
  DLOG(INFO) << "BatchJobQueue: put " << job << " into queue";
  lock.unlock();
  put_job_.notify_one();
  return true;
}

std::vector<std::shared_ptr<Job>> BatchJobQueue::GetBatch(
    size_t batch_size, size_t timeout_ms) {
  auto begin = std::chrono::steady_clock::now();
  std::vector<std::shared_ptr<Job>> batch_jobs;
  while (batch_jobs.size() < batch_size) {
    std::unique_lock<std::mutex> lock{mutex_};
    put_job_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                      [this] { return !this->queue_.empty(); });
    if (!queue_.empty()) {
      batch_jobs.push_back(queue_.front());
      DLOG(INFO) << "put " << queue_.front() << " into batch_jobs";
      queue_.pop();
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - begin).count();
    if (!batch_jobs.empty() && duration > timeout_ms) {
      break;
    }
  }
  CHECK(!batch_jobs.empty()) << "batch_jobs is empty";
  return batch_jobs;
}

}  // namespace colserve