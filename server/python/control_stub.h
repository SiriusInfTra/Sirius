#ifndef COLSERVE_CONTROL_STUB_H
#define COLSERVE_CONTROL_STUB_H

#include <memory>
#include <thread>
#include "../block_queue.h"

namespace pycolserve {
using namespace colserve;

enum class Event {
  // status event
  kTrainStart,
  kTrainEnd,
  kInterruptTrainDone,
  kResumeTrainDone,
  kColocateAdjustL1Done,

  // cmd event
  kInterruptTrain,
  kResumeTrain,
  kColocateAdjustL1,
};

class SwitchStub {
 public:
  SwitchStub();
  void Stop();
  int Cmd();
  void Cmd(int cmd);
  void TrainStart();
  void TrainEnd();

 private:
  bool running_{true};
  int cmd_{-1};
  std::unique_ptr<MemoryQueue<int>> cmd_event_mq_, status_event_mq_;
  std::unique_ptr<std::thread> thread_;
};

class ColocateStub {
 public:
  ColocateStub();
  void Stop();
  int Cmd();
  void Cmd(int cmd);
  void ColocateAdjustL1Done();

 private:
  bool running_{true};
  int cmd_{-1};
  std::unique_ptr<MemoryQueue<int>> cmd_event_mq_, status_event_mq_;
  std::unique_ptr<std::thread> thread_;
};

}

#endif