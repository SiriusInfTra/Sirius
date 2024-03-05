#include "logging_as_glog.h"
#include <signal.h>
#include "controller.h"
#include "config.h"
#include "model_train_store.h"

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
      ModelTrainStore::Get()->SetCurBatchSize(entry.value);
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
    if (ModelTrainStore::Get()->GetTrainPid() != -1) {
      LOG(INFO) << "[Controller]: kill train";
      CHECK_EQ((kill(ModelTrainStore::Get()->GetTrainPid(), SIGKILL)), 0);
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

uint64_t Controller::ColocateAdjust(size_t batch_size) {
  auto cmd_id = Controller::adjust_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    if (Config::serve_mode == ServeMode::kColocateL1) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1), static_cast<int>(batch_size)});
    } else if (Config::serve_mode == ServeMode::kColocateL2) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2), static_cast<int>(batch_size)});
    }
    ModelTrainStore::Get()->AddTargetBatchSize(-batch_size);
  }
  return cmd_id;
}

uint64_t Controller::InferExit(size_t batch_size) {
  static std::atomic<uint64_t> infer_exit_id = 1;
  auto cmd_id = infer_exit_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    if (Config::IsColocateMode()) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInferExit), static_cast<int>(batch_size)});
      ModelTrainStore::Get()->AddTargetBatchSize(batch_size);
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

bool Controller::TryEnterInferModelAlloc(size_t model_rank) {
  if (last_alloc_infer_model_ != static_cast<size_t>(-1)) {
    return false;
  }
  EnterInferModelAlloc(model_rank);
  return true;
}

void Controller::EnterInferModelAlloc(size_t model_rank) {
  std::unique_lock<std::mutex> lock{infer_model_alloc_mutex_};
  infer_model_alloc_cv_.wait(lock, [this, model_rank]() {
    return last_alloc_infer_model_ == static_cast<size_t>(-1);
  });
  last_alloc_infer_model_ = model_rank;
  LOG(INFO) << "InferModel " << model_rank << " enter allocation";
}

void Controller::ExitInferModelAlloc(size_t model_rank) {
  CHECK_EQ(last_alloc_infer_model_, model_rank);
  last_alloc_infer_model_ = static_cast<size_t>(-1);
  infer_model_alloc_cv_.notify_one();
  LOG(INFO) << "InferModel " << model_rank << " exit allocation";
}

} // namespace colserve
