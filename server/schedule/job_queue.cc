#include <server/logging_as_glog.h>
#include <server/grpc/grpc_server.h>
#include <server/schedule/job_queue.h>
#include <server/config.h>
#include <server/llm/llm_util.h>

#include <boost/json.hpp>
#include <chrono> 

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

LLMInferJob::LLMInferJob(network::InferHandler::InferData* data) 
    : InferJob(data) {
  CHECK_EQ(data->GetNumInputData(), 2); // prompt, sampling args

  prompt_ = std::string_view{data->GetInputData(0)};
  std::string_view boost_json_str{data->GetInputData(1)};
  auto json_value = boost::json::parse(boost_json_str);
  try {
    max_tokens_ = json_value.at("max_tokens").as_int64();
  } catch (const std::exception& e) {
    LOG(FATAL) << "LLMInferJob: " << e.what();
  }
}

void LLMInferJob::RecordProfile(const LLMRequestMetric &metric) {
  InferJob::RecordProfile();
  Profiler::Get()->RecordPerf(Profiler::PerfItem::LLMNumPromptTokens,
    metric.num_prompt_token);
  Profiler::Get()->RecordPerf(Profiler::PerfItem::LLMNumGenTokens, 
    metric.num_output_token);
  Profiler::Get()->RecordPerf(Profiler::PerfItem::LLMBackendQueue, 
    metric.queue_ms);
  Profiler::Get()->RecordPerf(Profiler::PerfItem::LLMPrefill, 
    metric.prefill_ms);
  if (metric.num_output_token > 1) {
    Profiler::Get()->RecordPerf(Profiler::PerfItem::LLMTimeBetweenTokens, 
      metric.decode_ms / (metric.num_output_token - 1));
  }
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
  job->RecordEnqueued();
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

size_t JobQueue::NumJobs() {
  std::unique_lock lock{mutex_};
  return queue_.size();
}

bool BatchJobQueue::Put(const std::shared_ptr<Job> &job) {
  std::unique_lock lock{mutex_};
  queue_.push(job);
  job->RecordEnqueued();
  DLOG_IF(INFO, Config::log_infer_sched) 
      << "BatchJobQueue: put " << job << " into queue";
  lock.unlock();
  put_job_cv_.notify_one();
  return true;
}

double BatchJobQueue::FirstJobQueueTime() {
  std::unique_lock lock{mutex_};
  if (queue_.empty()) {
    return 0;
  } else {
    return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - queue_.front()->GetEnQueueTime()).count();
  }
}

std::vector<std::shared_ptr<Job>> BatchJobQueue::GetBatch(
    size_t batch_size, size_t interval_ms, size_t timeout_ms) {
  std::vector<std::shared_ptr<Job>> batch_jobs;
  {
    std::unique_lock<std::mutex> lock{mutex_};
    // put_job_cv_.wait(lock, [this] { return !this->queue_.empty(); });
    auto has_job = put_job_cv_.wait_for(lock,
        std::chrono::milliseconds(timeout_ms), 
        [this] { return !this->queue_.empty(); });
    if (!has_job) {
      return {};
    }
    CHECK(!this->queue_.empty());
    batch_jobs.push_back(queue_.front());
    DLOG(INFO) << "put " << queue_.front() << " into batch_jobs";
    queue_.pop();
  }

  auto begin = std::chrono::steady_clock::now();
  std::chrono::microseconds timeout_us{interval_ms * 1000};
  std::chrono::microseconds rest_time_us{timeout_us};

  while (batch_jobs.size() < batch_size) {
    std::unique_lock<std::mutex> lock{mutex_};
    put_job_cv_.wait_for(lock, rest_time_us, 
                      [this] { return !this->queue_.empty(); });
    if (!queue_.empty()) {
      batch_jobs.push_back(queue_.front());
      VLOG(1) << "put " << queue_.front() << " into batch_jobs";
      queue_.pop();
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin);
    if (duration >= timeout_us) {
      break;
    } else {
      rest_time_us = timeout_us - duration;
    }
  }
  auto end = std::chrono::steady_clock::now();
  // DLOG(INFO) << "[BatchJobQueue]: batch interval " 
            //  << std::chrono::duration<double, std::milli>(end - begin).count() << " ms";
  CHECK(!batch_jobs.empty()) << "batch_jobs is empty";

  for (auto job : batch_jobs) { 
    job->RecordDequeued();
  }
  return batch_jobs;
}

}  // namespace colserve