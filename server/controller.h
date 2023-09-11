#ifndef COLSERVE_CONTROLLER_H
#define COLSERVE_CONTROLLER_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <glog/logging.h>

#include "block_queue.h"

namespace colserve {


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
  bool InterruptTrain();
  bool ResumeTrain();
  bool WaitTrainNotRunning();
  bool WaitInferIdle();

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

    // cmd event (switch mode)
    kInterruptTrain,
    kResumeTrain,
    kColocateAdjustL1,
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

  std::unique_ptr<MemoryQueue<int>> train_cmd_event_mq_, train_status_event_mq_;
  
  std::mutex wait_train_mutex_, wait_infer_mutex_;
  std::condition_variable wait_train_cv_, wait_infer_cv_;
  std::unique_ptr<std::thread> monitor_train_thread_;
  
 public:
  friend std::ostream& operator<<(std::ostream& os, const Controller::TrainStatus &status);
  
};

} // namespace 


#endif