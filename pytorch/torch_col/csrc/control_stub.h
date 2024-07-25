#ifndef TORCH_COL_CONTROL_STUB_H
#define TORCH_COL_CONTROL_STUB_H

#include <memory>
#include <thread>
#include <mutex>

#include <common/inf_tra_comm/communicator.h>


namespace torch_col {
using namespace colserve;


class StubBase {
 public:
  StubBase();
  virtual ~StubBase() = default;

  void Stop();
  int GetCmd();
  void SetCmd(int cmd);
  void TrainStart();
  void TrainEnd();
  void StepsNoInteruptBegin();
  void StepsNoInteruptEnd();
  virtual void ReportBatchSize(int batch_size);
  void EnableTorchColEngine();
  uint64_t GetTargetTpcMask();
  bool CanExitAfterInferWorkloadDone();

 protected:
  virtual void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) = 0;

  long infer_workload_done_timestamp_{0};
  int cmd_{-1};
  uint64_t cmd_id_{0};

  bool exec_step_{false};
  std::mutex step_mutex_;
  
  bool running_{true};
  std::unique_ptr<std::thread> thread_;

  std::mutex mutex_;
};


class DummyStub : public StubBase {
 public:
  DummyStub() : StubBase() {}

 protected:
  void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) override;
};

class SwitchStub: public StubBase {
 public:
  SwitchStub() : StubBase() {};
  bool TryInterruptTrainDone();
  // void Stop();
  // int Cmd() override;
  // void Cmd(int cmd);
  // void TrainStart();
  // void TrainEnd();
  // void ReportBatchSize(int batch_size);
  // void StepsNoInteruptBegin();
  // void StepsNoInteruptEnd();


  // bool CanExitAfterInferWorkloadDone() {
  //   return __CanExitAfterInferWorkloadDone(infer_workload_done_timestamp_);
  // }
 protected:
  void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) override;

 private:
  // long infer_workload_done_timestamp_{0};
  // bool running_{true};
  // int cmd_{-1};
  // uint64_t cmd_id_{0};
  uint64_t last_reply_cmd_id_{0};
  // bool exec_step_{false};
  // std::mutex cmd_mutex_; TODO: avoid concurrent cmd access
  // std::mutex mutex_;
  // std::unique_ptr<std::thread> thread_;
};

class ColocateStub: public StubBase {
 public:
  ColocateStub(int batch_size) 
      : StubBase(), target_bs_(batch_size), current_bs_(batch_size) {};

  // void Stop();
  // int Cmd() override;
  int GetTargetBatchSize();
  void ColocateAdjustL1Done();
  void ColocateAdjustL2Done();
  // void TrainStart();
  // void TrainEnd();
  double PassedTimeFromSetCmd();
  void ReportBatchSize(int batch_size) override;
  // void StepsNoInteruptBegin();
  // void StepsNoInteruptEnd();

  // bool CanExitAfterInferWorkloadDone() {
  //   return __CanExitAfterInferWorkloadDone(infer_workload_done_timestamp_);
  // }
 protected:
  void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) override;

 private:
  // long infer_workload_done_timestamp_{0};
  // bool running_{true};
  // int cmd_{-1};
  // uint64_t cmd_id_{0};
  int target_bs_, current_bs_;
  // std::mutex mutex_;
  // std::mutex step_mutex_;
  // bool exec_step_{false};

  std::chrono::time_point<std::chrono::steady_clock> set_cmd_time_;
  // std::unique_ptr<std::thread> thread_;
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