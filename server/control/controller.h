#ifndef COLSERVE_CONTROLLER_H
#define COLSERVE_CONTROLLER_H

#include <server/train_adjuster.h>

#include <common/inf_tra_comm/communicator.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>


namespace colserve {
namespace ctrl {

enum class InferStatus {
  kIdle,
  kRunning,
};

enum class TrainStatus {
  kIdle,
  kRunning,
  kInterrupted,
};

std::ostream& operator<<(std::ostream& os, const InferStatus &status);
std::ostream& operator<<(std::ostream& os, const TrainStatus &status);


class Controller {
 public:
  static void Init();
  static Controller* Get();

  Controller();
  uint64_t InterruptTrain();
  uint64_t ResumeTrain();
  uint64_t ColocateAdjust(size_t model_rank, int device_id, size_t batch_size);
  uint64_t ColocateInferRequireAdjust(
      size_t model_rank, int device_id, 
      const std::vector<TrainAdjuster::AdjustPlan> &adjust_plans);
  bool WaitTrainNotRunning();
  bool WaitTaskSwitchDone(uint64_t cmd_id);
  bool WaitInferIdle();
  bool WaitColocateAdjustDone(uint64_t cmd_id);
  // uint64_t InferExit(int device_id);
  uint64_t ColocateInferReleaseAdjust(const std::vector<TrainAdjuster::AdjustPlan> &adjust_plans);
  uint64_t DummyInferExit(int device_id, int target_batch_size);

  // void InferRequestInc(size_t inc=1);
  // void InferResponseInc(size_t inc=1);
  void SetInferStatus(InferStatus status);
  bool IsInferIdle();
  // void LogInferStatus();
  std::string GetInferStatusStr();

  void TrainStart();
  void TrainEnd();
  bool IsTrainIdle();


  bool HasFlyingColocateAdjust();

  void InferenceWorkloadDone();

 private:
  static std::unique_ptr<Controller> controller_;

  // cmd counter
  static std::atomic<uint64_t> adjust_cmd_id;
  static std::atomic<uint64_t> interrupt_cmd_id;
  static std::atomic<uint64_t> resume_cmd_id;
  static std::atomic<uint64_t> infer_exit_id;
  static std::atomic<uint64_t> dummy_infer_exit_id;
  static std::atomic<uint64_t> infer_workload_done_id;
  
  void TrainMonitor();

  InferStatus infer_status_{InferStatus::kIdle};
  TrainStatus train_status_{TrainStatus::kIdle};

  // switch mode
  std::mutex wait_train_mutex_,
             wait_task_switch_mutex_,
             wait_infer_mutex_;
  std::condition_variable wait_train_cv_, 
                          wait_task_switch_cv_,
                          wait_infer_cv_;
  uint64_t task_switch_done_id_{0};
  bool has_sent_resume_train_from_last_interrupt_{true};
  std::unique_ptr<std::thread> monitor_train_thread_;

  // colocate mode
  std::mutex wait_train_adjust_mutex_;
  std::condition_variable wait_train_adjust_cv_;
  uint64_t adjust_done_id_{0};
  
  // // constrol cooperated allocation
  // std::mutex infer_change_memory_mutex_; // to allocate inference model in a sequential way
  // std::condition_variable infer_change_memory_cv_;
  // std::atomic<size_t> last_infer_change_memory_model_ = static_cast<size_t>(-1);
};

}
}

#endif