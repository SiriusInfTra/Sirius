#ifndef TORCH_COL_CONTROL_STUB_H
#define TORCH_COL_CONTROL_STUB_H

#include <memory>
#include <thread>
#include <mutex>

#include <common/block_queue.h>
#include <common/controlling.h>


namespace torch_col {
using namespace colserve;

bool __CanExitAfterInferWorkloadDone(long infer_workload_done_timestamp);

class StubBase {
public:
  virtual int Cmd() = 0;
  virtual ~StubBase() = default;
  void EnableTorchColEngine();
};


class DummyStub {
 public:
  DummyStub();
  void Stop();
  void TrainStart();
  void TrainEnd();

  bool CanExitAfterInferWorkloadDone() {
    return __CanExitAfterInferWorkloadDone(infer_workload_done_timestamp_);
  }
 private:
  long infer_workload_done_timestamp_{0};
  std::unique_ptr<MemoryQueue<ctrl::CtrlMsgEntry>> cmd_event_mq_, status_event_mq_;
  bool running_{true};
  std::mutex mutex_;
  std::unique_ptr<std::thread> thread_;
};

class SwitchStub: public StubBase {
 public:
  SwitchStub();
  void Stop();
  int Cmd() override;
  void Cmd(int cmd);
  void TrainStart();
  void TrainEnd();
  bool TryInterruptTrainDone();
  void ReportBatchSize(int batch_size);
  void StepsNoInteruptBegin();
  void StepsNoInteruptEnd();


  bool CanExitAfterInferWorkloadDone() {
    return __CanExitAfterInferWorkloadDone(infer_workload_done_timestamp_);
  }
 private:
  long infer_workload_done_timestamp_{0};
  bool running_{true};
  int cmd_{-1};
  uint64_t cmd_id_{0};
  uint64_t last_reply_cmd_id_{0};
  bool exec_step_{false};
  std::unique_ptr<MemoryQueue<ctrl::CtrlMsgEntry>> cmd_event_mq_, status_event_mq_;
  // std::mutex cmd_mutex_; TODO: avoid concurrent cmd access
  std::mutex mutex_;
  std::unique_ptr<std::thread> thread_;
};

class ColocateStub: public StubBase {
 public:
  ColocateStub(int batch_size);
  void Stop();
  int Cmd() override;
  int TargetBatchSize();
  void ColocateAdjustL1Done();
  void ColocateAdjustL2Done();
  void TrainStart();
  void TrainEnd();
  double PassedTimeFromSetCmd();
  void ReportBatchSize(int batch_size);
  void StepsNoInteruptBegin();
  void StepsNoInteruptEnd();

  bool CanExitAfterInferWorkloadDone() {
    return __CanExitAfterInferWorkloadDone(infer_workload_done_timestamp_);
  }
 private:
  long infer_workload_done_timestamp_{0};
  bool running_{true};
  int cmd_{-1};
  uint64_t cmd_id_{0};
  int target_bs_, current_bs_;
  std::mutex mutex_;
  std::mutex step_mutex_;
  bool exec_step_{false};

  std::chrono::time_point<std::chrono::steady_clock> set_cmd_time_;
  std::unique_ptr<MemoryQueue<ctrl::CtrlMsgEntry>> cmd_event_mq_, status_event_mq_; // adjust_event_mq_;
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


}

#endif