#include <glog/logging.h>

#include "control_stub.h"

namespace pycolserve {

using namespace colserve;

SwitchStub::SwitchStub() {
  cmd_event_mq_ = std::make_unique<MemoryQueue<int>>("cmd-ctrl", false);
  status_event_mq_ = std::make_unique<MemoryQueue<int>>("status-ctrl", false);

  thread_.reset(new std::thread([&](){
    while (running_) {
      int data;
      bool succ = cmd_event_mq_->TimedGet(data, 1000);
      if (succ) {
        if (data == static_cast<int>(Event::kInterruptTrain)) {
          cmd_ = static_cast<int>(Event::kInterruptTrain);
          LOG(INFO) << "[SwitchStub] Interrupt train";
          status_event_mq_->Put(static_cast<int>(Event::kInterruptTrainDone));
        } else if (data == static_cast<int>(Event::kResumeTrain)) {
          cmd_ = static_cast<int>(Event::kResumeTrain);
          LOG(INFO) << "[SwitchStub] Resume train";
          status_event_mq_->Put(static_cast<int>(Event::kResumeTrainDone));
        } else {
          LOG(WARNING) << "[SwitchStub] Unknown command: " << data;
        }
      } else {
        // LOG(INFO) << "[SwitchStub] No command";
      }
    }
  }));
}

void SwitchStub::Stop() {
  running_ = false;
  thread_->join();
}

void SwitchStub::TrainStart() {
  status_event_mq_->Put(static_cast<int>(Event::kTrainStart));
}

void SwitchStub::TrainEnd() {
  status_event_mq_->Put(static_cast<int>(Event::kTrainEnd));
}

int SwitchStub::Cmd() {
  return cmd_;
}

void SwitchStub::Cmd(int cmd) {
  cmd_ = cmd;
}

ColocateStub::ColocateStub() {
  cmd_event_mq_ = std::make_unique<MemoryQueue<int>>("cmd-ctrl", false);
  status_event_mq_ = std::make_unique<MemoryQueue<int>>("status-ctrl", false);
  adjust_event_mq_ = std::make_unique<MemoryQueue<int>>("adjust-ctrl", false);

  thread_.reset(new std::thread([&]() {
    while (running_) {
      int data;
      bool succ = cmd_event_mq_->TimedGet(data, 1000);
      if (succ) {
        if (data == static_cast<int>(Event::kColocateAdjustL1)) {
          LOG(INFO) << "[ColocateStub] Adjust L1";
          cmd_ = static_cast<int>(Event::kColocateAdjustL1);
          set_cmd_time_ = std::chrono::steady_clock::now();
        } else if (data == static_cast<int>(Event::kColocateAdjustL2)) {
          LOG(INFO) << "[ColocateStub] Adjust L2";
          cmd_ = static_cast<int>(Event::kColocateAdjustL2);
          set_cmd_time_ = std::chrono::steady_clock::now();
        } else {
          LOG(WARNING) << "[ColocateStub] Unknown command: " << data;
        }
      } else {
        // LOG(INFO) << "[ColocateStub] No command";
      }
    }
  }));
}

void ColocateStub::Stop() {
  running_ = false;
  thread_->join();
}

int ColocateStub::Cmd() {
  return cmd_;
}

void ColocateStub::ColocateAdjustL1Done() {
  adjust_event_mq_->Put(static_cast<int>(Event::kColocateAdjustL1Done));
  LOG(INFO) << "[ColocateStub] Adjust L1 done";
}

void ColocateStub::ColocateAdjustL2Done() {
  adjust_event_mq_->Put(static_cast<int>(Event::kColocateAdjustL2Done));
  LOG(INFO) << "[ColocateStub] Adjust L2 done";
}

void ColocateStub::TrainStart() {
  status_event_mq_->Put(static_cast<int>(Event::kTrainStart));
}

void ColocateStub::TrainEnd() {
  status_event_mq_->Put(static_cast<int>(Event::kTrainEnd));
}

void ColocateStub::Cmd(int cmd) {
  cmd_ = cmd;
}

double ColocateStub::PassedTimeFromSetCmd() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - set_cmd_time_).count();
}

} // namespace pycolserve