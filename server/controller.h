#ifndef COLSERVE_CONTROLLER_H
#define COLSERVE_CONTROLLER_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <glog/logging.h>

#include "block_queue.h"

namespace colserve {

struct CtrlMsgEntry {
  uint64_t id;
  int event;
};

class Controller {
 public:
  static void Init();
  static Controller* Get() {
    if (controller_ == nullptr) {
      LOG(FATAL) << "Controller not initialized";
    }
    return controller_.get();
  }

  Controller();
  uint64_t InterruptTrain();
  uint64_t ResumeTrain();
  uint64_t ColocateAdjust();
  bool WaitTrainNotRunning();
  bool WaitInferIdle();
  bool WaitColocateAdjustDone(uint64_t cmd_id);
  uint64_t InferExit();

  void InferRequestInc(size_t inc=1);
  void InferResponseInc(size_t inc=1);
  bool IsInferIdle();

  void TrainStart();
  void TrainEnd();
  bool IsTrainIdle();

  enum class Event {
    // status event
    kTrainStart,
    kTrainEnd,
    kInterruptTrainDone,
    kResumeTrainDone,
    kColocateAdjustL1Done,
    kColocateAdjustL2Done,

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
  
  
 public:
  friend std::ostream& operator<<(std::ostream& os, const Controller::TrainStatus &status);
  
};

} // namespace 


#endif