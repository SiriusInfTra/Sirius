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
  void SetTrainFirstEpochDone();

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
  DummyStub() : StubBase() {
    DLOG(INFO) << "[DummyStub] initialized";
  }

 protected:
  void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) override;
};

class SwitchStub: public StubBase {
 public:
  SwitchStub();
  bool TryInterruptTrainDone(bool barrier);
  void SetKilledBatchRecover();
  void TrainResumeDone();
  bool PrepareResume();

  void SetGlobalInterruptFlag(bool flag);
  void SetGlobalHasBatchKilled(bool flag) {
    bip::scoped_lock lock{*shared_data_.mut_};
    *shared_data_.has_batch_killed_ = flag;
  }

  bool GetGlobalInterruptFlag();
  bool GetGlobalHasBatchKilled() {
    bip::scoped_lock lock{*shared_data_.mut_};
    return *shared_data_.has_batch_killed_;
  }

 protected:
  void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) override;
  bool TryInterruptTrainDoneWithLock(bool barrier);

 private:
  uint64_t last_reply_cmd_id_{0};
  std::atomic<bool> has_killed_batch_recover_{true};

  struct GlobalSharedData {
    bool *interrupt_flag_{nullptr};
    bool *has_batch_killed_{nullptr};
    bip_mutex *mut_;
  } shared_data_;
};

class ColocateStub: public StubBase {
 public:
  ColocateStub(int batch_size) 
      : StubBase(), 
        input_batch_size_(batch_size), 
        current_bs_(batch_size) {
    DLOG(INFO) << "[ColocateStub] initialized";
  };

  int GetTargetBatchSize();
  int GetUnpubTargetBatchSize();
  void ColocateAdjustL1Done();
  void ColocateAdjustL2Done();
  double PassedTimeFromSetCmd();
  void ReportBatchSize(int batch_size) override;

  void SetKilledBatchRecover() {
    has_killed_batch_recover_.store(true, std::memory_order_release);
  }
  void SetKilledBatchReconfiged() {
    will_killed_batch_reconfig_.store(false, std::memory_order_release);
  }

 protected:
  void ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) override;

 private:
  int input_batch_size_, current_bs_;
  std::atomic<bool> has_killed_batch_recover_{true};
  std::atomic<bool> will_killed_batch_reconfig_{false};
  std::chrono::time_point<std::chrono::steady_clock> set_cmd_time_;
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