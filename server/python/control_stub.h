#ifndef COLSERVE_CONTROL_STUB_H
#define COLSERVE_CONTROL_STUB_H

#include <memory>
#include <thread>

#include "../block_queue.h"
#include "../controller.h"

namespace pycolserve {
using namespace colserve;

enum class Event {
  // status event
  kTrainStart,
  kTrainEnd,
  kInterruptTrainDone,
  kResumeTrainDone,
  kColocateAdjustL1Done,
  kColocateAdjustL2Done,

  // cmd event
  kInterruptTrain,
  kResumeTrain,
  kColocateAdjustL1,
  kColocateAdjustL2,
  kInferExit,
};

class SwitchStub {
 public:
  SwitchStub();
  void Stop();
  int Cmd();
  void Cmd(int cmd);
  void TrainEnd();
  void TrainStart();

 private:
  bool running_{true};
  int cmd_{-1};
  std::unique_ptr<MemoryQueue<CtrlMsgEntry>> cmd_event_mq_, status_event_mq_;
  std::unique_ptr<std::thread> thread_;
};

class ColocateStub {
 public:
  ColocateStub(int batch_size);
  void Stop();
  int Cmd();
  int TargetBatchSize();
  void ColocateAdjustL1Done();
  void ColocateAdjustL2Done();
  void TrainStart();
  void TrainEnd();
  double PassedTimeFromSetCmd();

 private:
  bool running_{true};
  int cmd_{-1};
  uint64_t cmd_id_;
  int target_bs_, current_bs_;

  std::chrono::time_point<std::chrono::steady_clock> set_cmd_time_;
  std::unique_ptr<MemoryQueue<CtrlMsgEntry>> cmd_event_mq_, status_event_mq_; // adjust_event_mq_;
  std::unique_ptr<std::thread> thread_;
};

}

#endif