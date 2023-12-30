#ifndef COLSERVE_CONTROLLER_H
#define COLSERVE_CONTROLLER_H

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "block_queue.h"

namespace colserve {

struct CtrlMsgEntry {
  uint64_t id;
  int event;
  int value;
};

class Controller {
 public:
  static void Init();
  static Controller* Get();

  Controller();
  uint64_t InterruptTrain();
  uint64_t ResumeTrain();
  uint64_t ColocateAdjust(size_t batch_size);
  bool WaitTrainNotRunning();
  bool WaitInferIdle();
  bool WaitColocateAdjustDone(uint64_t cmd_id);
  uint64_t InferExit(size_t batch_size);

  void InferRequestInc(size_t inc=1);
  void InferResponseInc(size_t inc=1);
  bool IsInferIdle();
  void LogInferStatus();

  void TrainStart();
  void TrainEnd();
  bool IsTrainIdle();

  bool HasFlyingColocateAdjust();

  bool TryEnterInferModelAlloc(size_t model_rank);
  void EnterInferModelAlloc(size_t model_rank);
  void ExitInferModelAlloc(size_t model_rank);

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
  };

 private:
  static std::unique_ptr<Controller> controller_;
  struct InferStatus {
    enum {
      kIdle,
      kRunning,
    } status{kIdle};
    size_t num_requests{0};
    size_t num_responses{0};
    std::mutex mutex;
  };

  struct TrainStatus {
    enum {
      kIdle,
      kRunning,
      kInterrupted,
    } status{kIdle};
  };

  void MonitorTrain();

  InferStatus infer_status_;
  TrainStatus train_status_;

  // std::unique_ptr<MemoryQueue<int>> , train_adjust_event_mq_;
  std::unique_ptr<MemoryQueue<CtrlMsgEntry>> train_status_event_mq_, train_cmd_event_mq_;
  
  // switch mode
  std::mutex wait_train_mutex_, wait_infer_mutex_;
  std::condition_variable wait_train_cv_, wait_infer_cv_;
  std::unique_ptr<std::thread> monitor_train_thread_;

  // colocate mode
  std::mutex wait_train_adjust_mutex_;
  std::condition_variable wait_train_adjust_cv_;
  uint64_t adjust_done_id_{0};
  
  // constrol cooperated allocation
  std::mutex infer_model_alloc_mutex_; // to allocate inference model in a sequential way
  std::condition_variable infer_model_alloc_cv_;
  std::atomic<size_t> last_alloc_infer_model_ = static_cast<size_t>(-1);

  // cmd counter
  static std::atomic<uint64_t> adjust_cmd_id;

 public:
  friend std::ostream& operator<<(std::ostream& os, const Controller::TrainStatus &status);
  
};

} // namespace 


#endif