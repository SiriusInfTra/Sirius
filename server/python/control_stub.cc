#include <glog/logging.h>

#include "control_stub.h"

namespace pycolserve {

using namespace colserve;

SwitchStub::SwitchStub() {
  cmd_event_mq_ = std::make_unique<MemoryQueue<CtrlMsgEntry>>("cmd-ctrl", false);
  status_event_mq_ = std::make_unique<MemoryQueue<CtrlMsgEntry>>("status-ctrl", false);

  thread_.reset(new std::thread([&](){
    while (running_) {
      CtrlMsgEntry data;
      bool succ = cmd_event_mq_->TimedGet(data, 1000);
      if (succ) {
        if (data.event == static_cast<int>(Event::kInterruptTrain)) {
          cmd_ = static_cast<int>(Event::kInterruptTrain);
          LOG(INFO) << "[SwitchStub] Interrupt train";
          status_event_mq_->Put({0, static_cast<int>(Event::kInterruptTrainDone)});
        } else if (data.event == static_cast<int>(Event::kResumeTrain)) {
          cmd_ = static_cast<int>(Event::kResumeTrain);
          LOG(INFO) << "[SwitchStub] Resume train";
          status_event_mq_->Put({0, static_cast<int>(Event::kResumeTrainDone)});
        } else {
          LOG(WARNING) << "[SwitchStub] Unknown command: " << data.event;
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
  status_event_mq_->Put({0, static_cast<int>(Event::kTrainStart)});
}

void SwitchStub::TrainEnd() {
  status_event_mq_->Put({0, static_cast<int>(Event::kTrainEnd)});
}

int SwitchStub::Cmd() {
  return cmd_;
}

void SwitchStub::Cmd(int cmd) {
  cmd_ = cmd;
}

ColocateStub::ColocateStub(int batch_size) : target_bs_(batch_size), current_bs_(batch_size) {
  cmd_event_mq_ = std::make_unique<MemoryQueue<CtrlMsgEntry>>("cmd-ctrl", false);
  status_event_mq_ = std::make_unique<MemoryQueue<CtrlMsgEntry>>("status-ctrl", false);
  // adjust_event_mq_ = std::make_unique<MemoryQueue<int>>("adjust-ctrl", false);

  thread_.reset(new std::thread([&]() {
    while (running_) {
      CtrlMsgEntry data;
      bool succ = cmd_event_mq_->TimedGet(data, 1000);
      if (succ) {
        if (data.event == static_cast<int>(Event::kColocateAdjustL1)
            || data.event == static_cast<int>(Event::kColocateAdjustL2)) {
          this->target_bs_ -= 3;
          LOG(INFO) << "[ColocateStub] Adjust batch size, target " << this->target_bs_;
          CHECK_LT(this->target_bs_, this->current_bs_);
          cmd_ = data.event;
          cmd_id_ = data.id;
          set_cmd_time_ = std::chrono::steady_clock::now();
        } else if (data.event == static_cast<int>(Event::kInferExit)) {
          this->target_bs_ += 3;
          LOG(INFO) << "[ColocateStub] Infer Exit adjust back to " << this->target_bs_;
          CHECK_LE(this->target_bs_, this->current_bs_);
          if (this->target_bs_ == this->current_bs_) {
            if (cmd_ == static_cast<int>(Event::kColocateAdjustL1)) {
              this->ColocateAdjustL1Done();
            } else if (cmd_ == static_cast<int>(Event::kColocateAdjustL2)) {
              this->ColocateAdjustL2Done();
            }
          }
        } else {
          LOG(WARNING) << "[ColocateStub] Unknown command: " << data.event;
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

int ColocateStub::TargetBatchSize() {
  return target_bs_;
}

void ColocateStub::ColocateAdjustL1Done() {
  if (cmd_ == static_cast<int>(Event::kColocateAdjustL1)) {
    status_event_mq_->Put({cmd_id_, static_cast<int>(Event::kColocateAdjustL1Done)});
    cmd_ = -1;
    cmd_id_ = 0;
    LOG(INFO) << "[ColocateStub] Adjust L1 done";
  }
}

void ColocateStub::ColocateAdjustL2Done() {
  if (cmd_ == static_cast<int>(Event::kColocateAdjustL2)) {
    status_event_mq_->Put({cmd_id_, static_cast<int>(Event::kColocateAdjustL2Done)});
    cmd_ = -1;
    cmd_id_ = 0;
    LOG(INFO) << "[ColocateStub] Adjust L2 done";
  }
}

void ColocateStub::TrainStart() {
  status_event_mq_->Put({0, static_cast<int>(Event::kTrainStart)});
}

void ColocateStub::TrainEnd() {
  status_event_mq_->Put({0, static_cast<int>(Event::kTrainEnd)});
}

double ColocateStub::PassedTimeFromSetCmd() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - set_cmd_time_).count();
}

} // namespace pycolserve