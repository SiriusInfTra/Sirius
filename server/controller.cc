#include "logging_as_glog.h"
#include <common/util.h>
#include <server/train_launcher.h>
#include <server/resource_manager.h>
#include <server/infer_model_store.h>
#include <server/controller.h>
#include <server/config.h>

#include <signal.h>
#include <algorithm>


namespace colserve
{
  
std::unique_ptr<Controller> Controller::controller_;

std::atomic<uint64_t> Controller::adjust_cmd_id = 1;

std::ostream& operator<<(std::ostream&os, ctrl::CtrlEvent event) {
  switch (event) {
  // status event
  case ctrl::CtrlEvent::kTrainStart:
    os << "ctrl::CtrlEvent::kTrainStart"; 
    return os;
  case ctrl::CtrlEvent::kTrainEnd:
    os << "ctrl::CtrlEvent::kTrainEnd"; 
    return os;
  case ctrl::CtrlEvent::kInterruptTrainDone:
    os << "ctrl::CtrlEvent::kInterruptTrainDone";
    return os;
  case ctrl::CtrlEvent::kResumeTrainDone:
    os << "ctrl::CtrlEvent::kResumeTrainDone";
    return os;

  // cmd event
  case ctrl::CtrlEvent::kInterruptTrain:
    os << "ctrl::CtrlEvent::kInterruptTrain";
    return os;
  case ctrl::CtrlEvent::kResumeTrain:
    os << "ctrl::CtrlEvent::kResumeTrain";
    return os;
  default:
    LOG(FATAL) << "unknown ctrl::CtrlEvent";
    return os;
  }
}

std::ostream& operator<<(std::ostream& os, const Controller::TrainStatus &status) {
  switch (status.status) {
  case Controller::TrainStatus::kIdle:
    os << "Controller::TrainStatus::kIdle";
    return os;
  case Controller::TrainStatus::kRunning:
    os << "Controller::TrainStatus::kRunning";
    return os;
  case Controller::TrainStatus::kInterrupted:
    os << "Controller::TrainStatus::kInterrupted";
    return os;
  default:
    LOG(FATAL) << "unknown Controller::TrainStatus";
    return os;
  }
}

void Controller::Init() {
  controller_ = std::make_unique<Controller>();
}

Controller* Controller::Get(){
  if (controller_ == nullptr) {
    LOG(FATAL) << "Controller not initialized";
  }
  return controller_.get();
}

Controller::Controller() {
  train_cmd_event_mq_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("cmd-ctrl", true);
  train_status_event_mq_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("status-ctrl", true);
  // train_adjust_event_mq_ = std::make_unique<MemoryQueue<int>>("adjust-ctrl", true);
  monitor_train_thread_ = std::make_unique<std::thread>(&Controller::MonitorTrain, this);
}

void Controller::MonitorTrain() {
  while (true) {
    auto entry = static_cast<ctrl::CtrlMsgEntry>(train_status_event_mq_->BlockGet());
    auto event = static_cast<ctrl::CtrlEvent>(entry.event);
    auto cur_status = train_status_;
    // LOG(INFO) << "[Controller] MonitorTrain: " << event;
    switch (event){
    case ctrl::CtrlEvent::kTrainStart:
      CHECK_EQ(train_status_.status, TrainStatus::kIdle);
      train_status_.status = TrainStatus::kRunning;
      break;
    case ctrl::CtrlEvent::kInterruptTrainDone:
      train_status_.status = TrainStatus::kInterrupted;
      LOG(INFO) << "[Controller]: train interrupted";
      break;
    case ctrl::CtrlEvent::kResumeTrainDone:
      CHECK(train_status_.status == TrainStatus::kInterrupted 
          || train_status_.status == TrainStatus::kRunning);
      train_status_.status = TrainStatus::kRunning;
      break;
    case ctrl::CtrlEvent::kColocateAdjustL1Done:
    case ctrl::CtrlEvent::kColocateAdjustL2Done:
      adjust_done_id_ = entry.id;
      wait_train_adjust_cv_.notify_all();
      break;
    case ctrl::CtrlEvent::kTrainEnd:
      CHECK(train_status_.status == TrainStatus::kRunning
          || train_status_.status == TrainStatus::kIdle);
      train_status_.status = TrainStatus::kIdle;
      train_cmd_event_mq_->Clear();
      break;
    case ctrl::CtrlEvent::kReportBatchSize:
      CHECK(train_status_.status != TrainStatus::kIdle);
      TrainLauncher::Get()->SetCurBatchSize(entry.value);
      break;
    default:
      break;
    }
    if (cur_status.status == TrainStatus::kRunning 
        && train_status_.status != TrainStatus::kRunning) {
      wait_train_cv_.notify_all();
    }
    if (cur_status.status != train_status_.status) {
      LOG(INFO) << "[Controller] MonitorTrain: train status " 
                << cur_status << " -> " << train_status_;
    }
  }
}

uint64_t Controller::InterruptTrain() {
  static std::atomic<uint64_t> interrupt_cmd_id = 1;
  auto cmd_id = interrupt_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    if (train_status_.status == TrainStatus::kRunning) {
      // LOG(INFO) << "Controller: Put InterruptTrain";
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)});
    }
  } else if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    if (TrainLauncher::Get()->GetTrainPid() != -1) {
      LOG(INFO) << "[Controller]: kill train";
      CHECK_EQ((kill(TrainLauncher::Get()->GetTrainPid(), SIGKILL)), 0);
      // TrainEnd();
    }
  }
  return cmd_id;
}

uint64_t Controller::ResumeTrain() {
  static std::atomic<uint64_t> resume_cmd_id = 1;
  auto cmd_id = resume_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    // LOG(INFO) << "Controller: Put ResumeTrain";
    train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kResumeTrain)});
  }
  return cmd_id;
}

uint64_t Controller::ColocateAdjust(size_t model_rank, size_t batch_size) {
  auto cmd_id = Controller::adjust_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    if (Config::serve_mode == ServeMode::kColocateL1) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1), 
                                static_cast<int>(batch_size)});
    } else if (Config::serve_mode == ServeMode::kColocateL2) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2), 
                                static_cast<int>(batch_size)});
    }
    TrainLauncher::Get()->AddTargetBatchSize(-batch_size);
  }
  LOG(INFO) << "[Controller] model " << model_rank 
            << " send ColocateAdjust cmd_id: " << cmd_id 
            << " batch_size: " << batch_size
            << " train idle " << IsTrainIdle();
  return cmd_id;
}

uint64_t Controller::InferExit() {
  static std::atomic<uint64_t> infer_exit_id = 1;

  auto cmd_id = infer_exit_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    if (Config::IsColocateMode()) {
      // Controller::Get()->EnterInferChangeMemory();
      int train_target_bs;
      int train_bs;
      int train_bs_predict_by_avail_memory;
      double free_memory_MB;
      double reserve_memory_MB;
      double train_avail_memory_MB;
      {
        auto cold_cache_lock = ColdModelCache::Get().Lock();
        ResourceManager::InferMemoryChangingLock();
        // size_t reserve_memory_MB = sta::ByteToMB(Config::cold_cache_max_capability_nbytes - ColdModelCache::Get().GetCachedNbytes(cold_cache_lock));
        // train_avail_memory_MB = ResourceManager::GetTrainAvailMemoryMB() - reserve_memory_MB;
        // train_avail_memory_MB = std::max(train_avail_memory_MB, 0.0);
        // train_target_bs = TrainLauncher::Get()->PredictTargetBatchSize(train_avail_memory_MB);

        // min of bs predicted based on the actual and the predict
        train_avail_memory_MB = std::max(ResourceManager::GetTrainAvailMemoryMB(), 0.0);
        free_memory_MB = std::max(ResourceManager::GetFreeMemoryMB(), 0.0);
        reserve_memory_MB = sta::ByteToMB(Config::cold_cache_max_capability_nbytes 
                                          - ColdModelCache::Get().GetCachedNbytes(cold_cache_lock));
        auto adjust_bs = TrainLauncher::Get()->PredictTargetBatchSize(std::max(free_memory_MB - reserve_memory_MB, 0.0));
        train_bs_predict_by_avail_memory = TrainLauncher::Get()->PredictTargetBatchSize(
            std::max(train_avail_memory_MB - reserve_memory_MB, 0.0));
        train_bs = TrainLauncher::Get()->GetCurBatchSize();
        train_target_bs = std::min(train_bs + adjust_bs, train_bs_predict_by_avail_memory);

        TrainLauncher::Get()->SetTargetBatchSize(train_target_bs);
        ResourceManager::InferMemoryChangingUnlock();
        // TrainLauncher::Get()->AddTargetBatchSize(batch_size);
      }

      // LOG(INFO) << "[Controller] Infer Exit"
      //           << " train avail memory " << train_avail_memory_MB
      //           << " target batch size " << train_target_bs;
      LOG(INFO) << "[Controller] Infer Exit"
                << " cold cache reserve memory " << reserve_memory_MB
                << " train avail memory " << train_avail_memory_MB
                << " (predict bs " << train_bs_predict_by_avail_memory << ")"
                << " free memory " << free_memory_MB
                << " batch size " << train_bs << " -> " << train_target_bs;
      train_cmd_event_mq_->Put({cmd_id, 
                                static_cast<int>(ctrl::CtrlEvent::kInferExit), 
                                train_target_bs});
    }
  }
  return cmd_id;
}

bool Controller::WaitTrainNotRunning() {
  std::unique_lock lock{wait_train_mutex_};
  wait_train_cv_.wait(lock, [&]() { return train_status_.status != TrainStatus::kRunning; });
  return true;
}

bool Controller::WaitInferIdle() {
  std::unique_lock lock{wait_infer_mutex_};
  wait_infer_cv_.wait(lock, [&](){ return infer_status_.status == InferStatus::kIdle; });
  return true;
}

bool Controller::WaitColocateAdjustDone(uint64_t cmd_id) {
  if (!IsTrainIdle()) {
    if (Config::IsColocateMode()) {
      std::unique_lock lock{wait_train_adjust_mutex_};
      wait_train_adjust_cv_.wait(lock, [&](){ return adjust_done_id_ >= cmd_id; });
      // auto event = train_adjust_event_mq_->BlockGet();
    }
  }
  return true;
}

void Controller::InferRequestInc(size_t inc) {
  std::unique_lock lock{infer_status_.mutex};
  infer_status_.num_requests += inc;
  infer_status_.status = InferStatus::kRunning;
  // LOG(INFO) << "num_req: " <<infer_status_.num_requests 
  //           << " num_resp: " << infer_status_.num_responses;
}

void Controller::InferResponseInc(size_t inc) {
  std::unique_lock lock{infer_status_.mutex};
  infer_status_.num_responses += inc;
  if (infer_status_.num_responses == infer_status_.num_requests) {
    infer_status_.status = InferStatus::kIdle;
    wait_infer_cv_.notify_one();
    DLOG(INFO) << "[Controller] InferStatus -> kIdle";
  }
  // LOG(INFO) << "num_resp: " << infer_status_.num_responses
  //           << " num_req: " << infer_status_.num_requests;
}

bool Controller::IsInferIdle() {
  std::unique_lock lock{infer_status_.mutex};
  return infer_status_.status == InferStatus::kIdle;
}

void Controller::LogInferStatus() {
  std::unique_lock lock{infer_status_.mutex};
  LOG(INFO) << "[Controller] InferStatus: " << infer_status_.status
            << " num_req: " << infer_status_.num_requests
            << " num_resp: " << infer_status_.num_responses;
}

std::string Controller::GetInferStatusStr() {
  std::stringstream ss;
  std::unique_lock lock{infer_status_.mutex};
  ss << "InferStatus: " << infer_status_.status
     << " num_req: " << infer_status_.num_requests
     << " num_resp: " << infer_status_.num_responses;
  return ss.str();
}

void Controller::TrainStart() {
  // LOG(INFO) << "Controller Put TrainStart";
  train_status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)});
}

void Controller::TrainEnd() {
  train_status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)});
}

bool Controller::IsTrainIdle() {
  return train_status_.status == TrainStatus::kIdle;
}

bool Controller::HasFlyingColocateAdjust() {
  return adjust_done_id_ + 1 < Controller::adjust_cmd_id;
}



} // namespace colserve
