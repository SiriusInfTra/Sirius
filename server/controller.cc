#include <signal.h>

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

Controller::Controller() {
  train_cmd_event_mq_ = std::make_unique<MemoryQueue<int>>("cmd-ctrl", true);
  train_status_event_mq_ = std::make_unique<MemoryQueue<int>>("status-ctrl", true);
  train_adjust_event_mq_ = std::make_unique<MemoryQueue<int>>("adjust-ctrl", true);
  monitor_train_thread_ = std::make_unique<std::thread>(&Controller::MonitorTrain, this);
}

void Controller::MonitorTrain() {
  while (true) {
    auto event = static_cast<Event>(train_status_event_mq_->BlockGet());
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

bool Controller::InterruptTrain() {
  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    if (!IsTrainIdle()) {
      // LOG(INFO) << "Controller: Put InterruptTrain";
      train_cmd_event_mq_->Put(static_cast<int>(Event::kInterruptTrain));
    }
  } else if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    if (ModelTrainStore::Get()->GetTrainPid() != -1) {
      LOG(INFO) << "[Controller]: kill train";
      CHECK_EQ((kill(ModelTrainStore::Get()->GetTrainPid(), SIGKILL)), 0);
      // TrainEnd();
    }
  }
  return true;
}

bool Controller::ResumeTrain() {
  if (!IsTrainIdle()) {
    // LOG(INFO) << "Controller: Put ResumeTrain";
    train_cmd_event_mq_->Put(static_cast<int>(Event::kResumeTrain));
  }
  return true;
}

bool Controller::ColocateAdjust() {
  if (!IsTrainIdle()) {
    if (Config::serve_mode == ServeMode::kColocateL2) {
      train_cmd_event_mq_->Put(static_cast<int>(Event::kColocateAdjustL2));
    }
  }
  return true;
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

bool Controller::WaitColocateAdjustDone() {
  if (!IsTrainIdle()) {
    if (Config::serve_mode == ServeMode::kColocateL2) {
      auto event = train_adjust_event_mq_->BlockGet();
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
  train_status_event_mq_->Put(static_cast<int>(Event::kTrainStart));
}

void Controller::TrainEnd() {
  train_status_event_mq_->Put(static_cast<int>(Event::kTrainEnd));
}

bool Controller::IsTrainIdle() {
  return train_status_.status == TrainStatus::kIdle;
}

} // namespace colserve
