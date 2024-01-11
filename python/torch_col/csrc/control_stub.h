#ifndef COLSERVE_CONTROL_STUB_H
#define COLSERVE_CONTROL_STUB_H

#include <memory>
#include <thread>
#include <mutex>

// #include <block_queue.h>
// #include <controller.h>
#include <server/block_queue.h>
#include <server/controller.h>
// #include "../../server/controller.h"

extern int KillBatchOnRecv;

namespace torch_col {
using namespace colserve;

enum class Event {
  // status event
  kTrainStart,
  kTrainEnd,
  kInterruptTrainDone,
  kResumeTrainDone,
  kColocateAdjustL1Done,
  kColocateAdjustL2Done,
  
  kReportBatchSize,

  // cmd event: switch mode
  kInterruptTrain,
  kResumeTrain,
  // cmd event: colocate mode
  kColocateAdjustL1,
  kColocateAdjustL2,
  kInferExit, // train adjust back

  kNumEvent,
};

class SwitchStub {
 public:
  SwitchStub();
  void Stop();
  int Cmd();
  void Cmd(int cmd);
  void TrainStart();
  void TrainEnd();
  bool TryInterruptTrainDone();
  void ReportBatchSize(int batch_size);
  void StepsNoInteruptBegin();
  void StepsNoInteruptEnd();

 private:
  bool running_{true};
  int cmd_{-1};
  uint64_t cmd_id_{0};
  uint64_t last_reply_cmd_id_{0};
  bool exec_step_{false};
  std::unique_ptr<MemoryQueue<CtrlMsgEntry>> cmd_event_mq_, status_event_mq_;
  // std::mutex cmd_mutex_; TODO: avoid concurrent cmd access
  std::mutex mutex_;
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
  void ReportBatchSize(int batch_size);
  void StepsNoInteruptBegin();
  void StepsNoInteruptEnd();

 private:
  bool running_{true};
  int cmd_{-1};
  uint64_t cmd_id_{0};
  int target_bs_, current_bs_;
  std::mutex mutex_;
  std::mutex step_mutex_;
  bool exec_step_{false};

  std::chrono::time_point<std::chrono::steady_clock> set_cmd_time_;
  std::unique_ptr<MemoryQueue<CtrlMsgEntry>> cmd_event_mq_, status_event_mq_; // adjust_event_mq_;
  std::unique_ptr<std::thread> thread_;
};

class StubProfiler {
 public:
  static std::vector<long> GetAdjustRequestTimeStamp() {
    std::unique_lock lock{StubProfiler::mutex_};
    return adjust_request_time_stamp_;
  }

  static std::vector<long> GetAdjustDoneTimeStamp() {
    std::unique_lock lock{StubProfiler::mutex_};
    return adjsut_done_time_stamp_;
  }

  static void RecordAdjustRequest();
  static void RecordAdjustDone();

 private:
  static std::vector<long> adjust_request_time_stamp_;
  static std::vector<long> adjsut_done_time_stamp_;
  static std::mutex mutex_;
};

void DumpMempoolFreeList(std::string filename);
void DumpMempoolBlockList(std::string filename);

}

#endif