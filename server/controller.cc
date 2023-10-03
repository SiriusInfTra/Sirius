#include <signal.h>
#include <glog/logging.h>

#include "controller.h"
#include "config.h"
#include "model_train_store.h"

namespace colserve
{
  
std::unique_ptr<Controller> Controller::controller_;

std::ostream& operator<<(std::ostream&os, Controller::Event event) {
  switch (event) {
  // status event
  case Controller::Event::kTrainStart:
    os << "Controller::Event::kTrainStart"; 
    return os;
  case Controller::Event::kTrainEnd:
    os << "Controller::Event::kTrainEnd"; 
    return os;
  case Controller::Event::kInterruptTrainDone:
    os << "Controller::Event::kInterruptTrainDone";
    return os;
  case Controller::Event::kResumeTrainDone:
    os << "Controller::Event::kResumeTrainDone";
    return os;

  // cmd event
  case Controller::Event::kInterruptTrain:
    os << "Controller::Event::kInterruptTrain";
    return os;
  case Controller::Event::kResumeTrain:
    os << "Controller::Event::kResumeTrain";
    return os;
  default:
    LOG(FATAL) << "unknown Controller::Event";
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
  train_cmd_event_mq_ = std::make_unique<MemoryQueue<CtrlMsgEntry>>("cmd-ctrl", true);
  train_status_event_mq_ = std::make_unique<MemoryQueue<CtrlMsgEntry>>("status-ctrl", true);
  // train_adjust_event_mq_ = std::make_unique<MemoryQueue<int>>("adjust-ctrl", true);
  monitor_train_thread_ = std::make_unique<std::thread>(&Controller::MonitorTrain, this);
}

void Controller::MonitorTrain() {
  while (true) {
    auto entry = static_cast<CtrlMsgEntry>(train_status_event_mq_->BlockGet());
    auto event = static_cast<Event>(entry.event);
    auto cur_status = train_status_;
    // LOG(INFO) << "[Controller] MonitorTrain: " << event;
    switch (event){
    case Event::kTrainStart:
      CHECK_EQ(train_status_.status, TrainStatus::kIdle);
      train_status_.status = TrainStatus::kRunning;
      break;
    case Event::kInterruptTrainDone:
      train_status_.status = TrainStatus::kInterrupted;
      LOG(INFO) << "[Controller]: train interrupted";
      break;
    case Event::kResumeTrainDone:
      CHECK(train_status_.status == TrainStatus::kInterrupted 
          || train_status_.status == TrainStatus::kRunning);
      train_status_.status = TrainStatus::kRunning;
      break;
    case Event::kColocateAdjustL2Done:
      adjust_done_id_ = entry.id;
      wait_train_adjust_cv_.notify_all();
      break;
    case Event::kTrainEnd:
      CHECK(train_status_.status == TrainStatus::kRunning
          || train_status_.status == TrainStatus::kIdle);
      train_status_.status = TrainStatus::kIdle;
      train_cmd_event_mq_->Clear();
      break;
    default:
      break;
    }
    if (cur_status.status == TrainStatus::kRunning 
        && train_status_.status != TrainStatus::kRunning) {
      wait_train_cv_.notify_one();
    }
    LOG(INFO) << "[Controller] MonitorTrain: train status " 
              << cur_status << " -> " << train_status_;
  }
}

uint64_t Controller::InterruptTrain() {
  static std::atomic<uint64_t> interrupt_cmd_id = 1;
  auto cmd_id = interrupt_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    if (!IsTrainIdle()) {
      // LOG(INFO) << "Controller: Put InterruptTrain";
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(Event::kInterruptTrain)});
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
    train_cmd_event_mq_->Put({cmd_id, static_cast<int>(Event::kResumeTrain)});
  }
  return cmd_id;
}

uint64_t Controller:: ColocateAdjust() {
  static std::atomic<uint64_t> adjust_cmd_id = 1;
  auto cmd_id = adjust_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    if (Config::serve_mode == ServeMode::kColocateL2) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(Event::kColocateAdjustL2)});
    }
  }
  return cmd_id;
}

uint64_t Controller::InferExit() {
  static std::atomic<uint64_t> infer_exit_id = 1;
  auto cmd_id = infer_exit_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    if (Config::serve_mode == ServeMode::kColocateL2) {
      train_cmd_event_mq_->Put({cmd_id, static_cast<int>(Event::kInferExit)});
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
    if (Config::serve_mode == ServeMode::kColocateL2) {
      std::unique_lock lock{wait_train_adjust_mutex_};
      wait_train_adjust_cv_.wait(lock, [&](){ return adjust_done_id_ >= cmd_id; });
      // auto event = train_adjust_event_mq_->BlockGet();
    }
  }
  return true;
}

void Controller::InferRequestInc(size_t inc) {
  infer_status_.num_requests += inc;
  infer_status_.status = InferStatus::kRunning;
  // LOG(INFO) << "num_req: " <<infer_status_.num_requests 
  //           << " num_resp: " << infer_status_.num_responses;
}

void Controller::InferResponseInc(size_t inc) {
  infer_status_.num_responses += inc;
  if (infer_status_.num_responses == infer_status_.num_requests) {
    infer_status_.status = InferStatus::kIdle;
    wait_infer_cv_.notify_one();
    LOG(INFO) << "[Controller] InferStatus -> kIdle";
  }
  // LOG(INFO) << "num_resp: " << infer_status_.num_responses
  //           << " num_req: " << infer_status_.num_requests;
}

bool Controller::IsInferIdle() {
  return infer_status_.status == InferStatus::kIdle;
}

void Controller::TrainStart() {
  // LOG(INFO) << "Controller Put TrainStart";
  train_status_event_mq_->Put({0, static_cast<int>(Event::kTrainStart)});
}

void Controller::TrainEnd() {
  train_status_event_mq_->Put({0, static_cast<int>(Event::kTrainEnd)});
}

bool Controller::IsTrainIdle() {
  return train_status_.status == TrainStatus::kIdle;
}

} // namespace colserve
