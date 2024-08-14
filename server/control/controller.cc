#include <server/logging_as_glog.h>
#include <server/train_launcher.h>
#include <server/resource_manager.h>
#include <server/model_store/infer_model_store.h>
#include <server/model_store/model_cache.h>
#include <server/control/controller.h>
#include <server/config.h>

#include <common/inf_tra_comm/communicator.h>
#include <common/util.h>

#include <signal.h>
#include <algorithm>


namespace colserve {
namespace ctrl {
  
std::ostream& operator<<(std::ostream &os, const InferStatus &status) {
  switch (status) {
  case InferStatus::kIdle:
    os << "InferStatus::kIdle";
    return os;
  case InferStatus::kRunning:
    os << "InferStatus::kRunning";
    return os;
  default:
    LOG(FATAL) << "unknown InferStatus " << static_cast<int>(status);
    return os;
  }
}

std::ostream& operator<<(std::ostream &os, const TrainStatus &status) {
  switch (status) {
  case TrainStatus::kIdle:
    os << "TrainStatus::kIdle";
    return os;
  case TrainStatus::kRunning:
    os << "TrainStatus::kRunning";
    return os;
  case TrainStatus::kInterrupted:
    os << "TrainStatus::kInterrupted";
    return os;
  default:
    LOG(FATAL) << "unknown TrainStatus " << static_cast<int>(status);
    return os;
  }
}


std::unique_ptr<Controller> Controller::controller_ = nullptr;

std::atomic<uint64_t> Controller::adjust_cmd_id = 1;
std::atomic<uint64_t> Controller::interrupt_cmd_id = 1;
std::atomic<uint64_t> Controller::resume_cmd_id = 1;
std::atomic<uint64_t> Controller::infer_exit_id = 1;
std::atomic<uint64_t> Controller::dummy_infer_exit_id = 1;
std::atomic<uint64_t> Controller::infer_workload_done_id = 1;


void Controller::Init() {
  controller_ = std::make_unique<Controller>();
}

Controller* Controller::Get(){
  if (controller_ == nullptr) {
    LOG(FATAL) << "Controller not initialized";
  }
  return controller_.get();
}

Controller::Controller() {
  InfTraCommunicator::Init(true, true, 
                           sta::DeviceManager::GetNumVisibleGpu());

  monitor_train_thread_ = std::make_unique<std::thread>(&Controller::TrainMonitor, this);
}

void Controller::TrainMonitor() {
  while (true) {
    // auto entry = static_cast<ctrl::CtrlMsgEntry>(train_status_mqs_->BlockGet());
    // auto entry = train_status_mqs_[0]->BlockGet();
    auto [train_id, entry] = InfTraCommunicator::GetMQ()
        ->BlockGetFromAny(InfTraMessageQueue::Direction::kTra2Inf);

    auto event = static_cast<ctrl::CtrlEvent>(entry.event);
    auto last_status = train_status_;
    // LOG(INFO) << "[Controller] MonitorTrain: " << event;
    switch (event){
    case ctrl::CtrlEvent::kTrainStart:
      {
        CHECK_EQ(train_status_, TrainStatus::kIdle);
        train_status_ = TrainStatus::kRunning;
      }
      break;
    case ctrl::CtrlEvent::kInterruptTrainDone:
      {
        train_status_ = TrainStatus::kInterrupted;
        LOG(INFO) << "[Controller]: train interrupted";
      }
      break;
    case ctrl::CtrlEvent::kResumeTrainDone:
      {
        CHECK(train_status_ == TrainStatus::kInterrupted 
              || train_status_ == TrainStatus::kRunning);
        train_status_ = TrainStatus::kRunning;
      }
      break;
    case ctrl::CtrlEvent::kColocateAdjustL1Done:
    case ctrl::CtrlEvent::kColocateAdjustL2Done:
      {
        adjust_done_id_ = entry.id;
        wait_train_adjust_cv_.notify_all();
      }
      break;
    case ctrl::CtrlEvent::kTrainEnd:
      {
        CHECK(train_status_ == TrainStatus::kRunning
            || train_status_ == TrainStatus::kIdle);
        train_status_ = TrainStatus::kIdle;
        InfTraCommunicator::GetMQ()->Clear();
      }
      break;
    case ctrl::CtrlEvent::kReportBatchSize:
      CHECK(train_status_ != TrainStatus::kIdle);
      TrainLauncher::Get()->SetCurBatchSize(entry.value);
      break;
    default:
      break;
    }
    if (last_status == TrainStatus::kRunning 
        && train_status_ != TrainStatus::kRunning) {
      wait_train_cv_.notify_all();
    }
    if (last_status != train_status_) {
      LOG(INFO) << "[Controller] MonitorTrain: train status " 
                << last_status << " -> " << train_status_;
    }
  }
}

uint64_t Controller::InterruptTrain() {
  auto cmd_id = Controller::interrupt_cmd_id.fetch_add(
      1, std::memory_order_relaxed);

  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    if (train_status_ == TrainStatus::kRunning) {
      DLOG(INFO) << "Controller: Put InterruptTrain";
      InfTraCommunicator::GetMQ()->PutAll(
          {cmd_id, static_cast<int>(CtrlEvent::kInterruptTrain)}, 
          InfTraMessageQueue::Direction::kInf2Tra);
    }
  } else if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    if (TrainLauncher::Get()->GetTrainPid() != -1) {
      DLOG(INFO) << "[Controller]: kill train";
      CHECK_EQ((kill(TrainLauncher::Get()->GetTrainPid(), SIGKILL)), 0);
      // TrainEnd();
    }
  }
  return cmd_id;
}

uint64_t Controller::ResumeTrain() {
  auto cmd_id = Controller::resume_cmd_id.fetch_add(
      1, std::memory_order_relaxed);

  if (!IsTrainIdle()) {
    DLOG(INFO) << "Controller: Put ResumeTrain";
    InfTraCommunicator::GetMQ()->PutAll(
        {cmd_id, static_cast<int>(CtrlEvent::kResumeTrain)}, 
        InfTraMessageQueue::Direction::kInf2Tra);
  }
  return cmd_id;
}

uint64_t Controller::ColocateAdjust(size_t model_rank, int device_id, size_t batch_size) {
  static std::mutex adjust_batch_mutex; // for quick fix concurrency

  auto cmd_id = Controller::adjust_cmd_id.fetch_add(
      1, std::memory_order_relaxed);

  if (!IsTrainIdle()) {
#if ADJUST_WITH_FLYING
    std::lock_guard lock{adjust_batch_mutex};
#endif
    TrainLauncher::Get()->AddTargetBatchSize(-batch_size);
    auto msg = CtrlMsgEntry{
      .id = cmd_id,
      .event = Config::serve_mode == ServeMode::kColocateL1
               ? static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)
               : static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2),
      .value = static_cast<int>(TrainLauncher::Get()->GetTargetBatchSize())
    };

    InfTraCommunicator::GetMQ()->PutAll(
        msg, InfTraMessageQueue::Direction::kInf2Tra);
  }
  LOG(INFO) << "[Controller] model " << model_rank 
            << " send ColocateAdjust cmd_id: " << cmd_id 
            << " batch_size: " << batch_size
            << " train idle " << IsTrainIdle();
  return cmd_id;
}

uint64_t Controller::InferExit(int device_id) {
  auto cmd_id = Controller::infer_exit_id.fetch_add(
      1, std::memory_order_relaxed);

  if (!IsTrainIdle()) {
    if (Config::IsColocateMode()) {
      // Controller::Get()->EnterInferChangeMemory();
      int train_target_bs;
      int cur_train_target_bs;
      int train_bs_predict_by_avail_memory;
      double free_memory_MB;
      double reserve_memory_MB;
      double train_avail_memory_MB;
      {
        auto cold_cache_lock = ColdModelCache::Get(0)->Lock();
        ResourceManager::InferMemoryChangingLock(device_id);
        // size_t reserve_memory_MB = sta::ByteToMB(Config::cold_cache_max_capability_nbytes - ColdModelCache::Get().GetCachedNbytes(cold_cache_lock));
        // train_avail_memory_MB = ResourceManager::GetTrainAvailMemoryMB() - reserve_memory_MB;
        // train_avail_memory_MB = std::max(train_avail_memory_MB, 0.0);
        // train_target_bs = TrainLauncher::Get()->PredictTargetBatchSize(train_avail_memory_MB);

        // min of bs predicted based on the actual and the predict
        train_avail_memory_MB = std::max(ResourceManager::GetTrainAvailMemoryMB(device_id, false), 0.0);
        free_memory_MB = std::max(ResourceManager::GetFreeMemoryMB(device_id, true), 0.0);
        reserve_memory_MB = ColdModelCache::Get(0)->GetReleaseReserveMemoryMB(cold_cache_lock);
        if (Config::use_shared_tensor) {
          auto adjust_bs = TrainLauncher::Get()->GetAdjustBatchSize(
              std::max(free_memory_MB - reserve_memory_MB, 0.0));
          train_bs_predict_by_avail_memory = TrainLauncher::Get()->PredictTargetBatchSize(
              std::max(train_avail_memory_MB - reserve_memory_MB, 0.0));
          cur_train_target_bs = TrainLauncher::Get()->GetTargetBatchSize();
          train_target_bs = std::max(
              cur_train_target_bs, 
              std::min(cur_train_target_bs + adjust_bs, train_bs_predict_by_avail_memory)
          );
        } else {
          // not use availd memory, resulting in wrong prediction
          train_bs_predict_by_avail_memory = -1;
          auto adjust_bs = TrainLauncher::Get()->GetAdjustBatchSize(
              std::max(free_memory_MB, 0.0));
          cur_train_target_bs = TrainLauncher::Get()->GetTargetBatchSize();
          train_target_bs = std::max(cur_train_target_bs, 
                                     cur_train_target_bs + adjust_bs);
        }

        TrainLauncher::Get()->SetTargetBatchSize(train_target_bs);
        ResourceManager::InferMemoryChangingUnlock(device_id);
        // TrainLauncher::Get()->AddTargetBatchSize(batch_size);
      }

      // LOG(INFO) << "[Controller] Infer Exit"
      //           << " train avail memory " << train_avail_memory_MB
      //           << " target batch size " << train_target_bs;
      std::stringstream ss;
      ss << "[Controller] Infer Exit"
         << " cold cache reserve memory " << reserve_memory_MB
         << " train avail memory " << train_avail_memory_MB;
      if (Config::use_shared_tensor) {
        ss << " (predict bs " << train_bs_predict_by_avail_memory << ")";
      } else {
        ss << " (not use avail memory to predict bs)";
      }
      ss << " free memory " << free_memory_MB
         << " target batch size " << cur_train_target_bs << " -> " << train_target_bs;
      LOG_IF(INFO, Config::log_controller) << ss.str();

      // LOG(INFO) << "[Controller] Infer Exit"
      //           << " cold cache reserve memory " << reserve_memory_MB
      //           << " train avail memory " << train_avail_memory_MB
      //           << " (predict bs " << train_bs_predict_by_avail_memory << ")"
      //           << " free memory " << free_memory_MB
      //           << " target batch size " << cur_train_target_bs << " -> " << train_target_bs;
      InfTraCommunicator::GetMQ()->PutAll(
          CtrlMsgEntry{
            .id=cmd_id, 
            .event=static_cast<int>(CtrlEvent::kInferExit), 
            .value=train_target_bs
          }, 
          InfTraMessageQueue::Direction::kInf2Tra);
    }
  }
  return cmd_id;
}

uint64_t Controller::DummyInferExit(int device_id, int target_batch_size) {
  CHECK(Config::dummy_adjust) << "only used for dummy adjust";

  auto cmd_id = Controller::dummy_infer_exit_id.fetch_add(
      1, std::memory_order_relaxed);

  if (!IsTrainIdle()) {
    InfTraCommunicator::GetMQ()->PutAll(
        CtrlMsgEntry{
          .id = cmd_id,
          .event = static_cast<int>(CtrlEvent::kInferExit),
          .value = target_batch_size
        }, 
        InfTraMessageQueue::Direction::kInf2Tra);
  }
  return cmd_id;
}

bool Controller::WaitTrainNotRunning() {
  std::unique_lock lock{wait_train_mutex_};
  wait_train_cv_.wait(lock, 
      [&]() { return train_status_ != TrainStatus::kRunning; });
  return true;
}

bool Controller::WaitInferIdle() {
  std::unique_lock lock{wait_infer_mutex_};
  wait_infer_cv_.wait(lock, 
      [&](){ return infer_status_ == InferStatus::kIdle; });
  return true;
}

bool Controller::WaitColocateAdjustDone(uint64_t cmd_id) {
  if (!IsTrainIdle()) {
    if (Config::IsColocateMode()) {
      std::unique_lock lock{wait_train_adjust_mutex_};
      wait_train_adjust_cv_.wait(lock, 
          [&](){ return adjust_done_id_ >= cmd_id; });
    }
  }
  return true;
}

// void Controller::InferRequestInc(size_t inc) {
//   // std::unique_lock lock{infer_status_.mutex};
//   // infer_status_.num_requests += inc;
//   infer_status_ = InferStatus::kRunning;
//   // LOG(INFO) << "num_req: " <<infer_status_.num_requests 
//   //           << " num_resp: " << infer_status_.num_responses;
// }

// void Controller::InferResponseInc(size_t inc) {
//   LOG(FATAL) << "to re-implemente";

//   // std::unique_lock lock{infer_status_.mutex};
//   // infer_status_.num_responses += inc;
//   // if (infer_status_.num_responses == infer_status_.num_requests) {
//   //   infer_status_.status = InferStatus::kIdle;
//   //   wait_infer_cv_.notify_one();
//   //   DLOG(INFO) << "[Controller] InferStatus -> kIdle";
//   // }
// }

void Controller::SetInferStatus(InferStatus status) {
  infer_status_ = status;
  if (status == InferStatus::kIdle) {
    wait_infer_cv_.notify_one();
    DLOG(INFO) << "[Controller] InferStatus -> kIdle";
  }
}

bool Controller::IsInferIdle() {
  return infer_status_ == InferStatus::kIdle;
}

// void Controller::LogInferStatus() {
//   LOG(FATAL) << "deprecated";

//   // std::unique_lock lock{infer_status_.mutex};
//   // LOG(INFO) << "[Controller] InferStatus: " << infer_status_.status
//   //           << " num_req: " << infer_status_.num_requests
//   //           << " num_resp: " << infer_status_.num_responses;
// }

std::string Controller::GetInferStatusStr() {
  LOG(FATAL) << "deprecated";
  
  // std::stringstream ss;
  // std::unique_lock lock{infer_status_.mutex};
  // ss << "InferStatus: " << infer_status_.status
  //    << " num_req: " << infer_status_.num_requests
  //    << " num_resp: " << infer_status_.num_responses;
  // return ss.str();
}

void Controller::TrainStart() {
  // LOG(INFO) << "Controller Put TrainStart";
  // train_status_mqs_[0]->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)});
  InfTraCommunicator::GetMQ()->Put(
      {0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)}, 
      InfTraMessageQueue::Direction::kTra2Inf,
      0);
}

void Controller::TrainEnd() {
  // train_status_mqs_[0]->Put(
  //     {0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)});
  InfTraCommunicator::GetMQ()->Put(
      {0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)}, 
      InfTraMessageQueue::Direction::kTra2Inf,
      0);
}

bool Controller::IsTrainIdle() {
  return train_status_ == TrainStatus::kIdle;
}

bool Controller::HasFlyingColocateAdjust() {
  return adjust_done_id_ + 1 < Controller::adjust_cmd_id;
}

void Controller::InferenceWorkloadDone() {
  auto cmd_id = Controller::infer_workload_done_id.fetch_add(
      1, std::memory_order_relaxed);

  if (Config::dummy_adjust) {
    LOG(INFO) << "[Controller] skip send InferenceWorkloadDone to train to eval dummy adjust";
    return;
  }

  InfTraCommunicator::GetMQ()->PutAll(
      {cmd_id, static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)}, 
      InfTraMessageQueue::Direction::kInf2Tra);

  LOG(INFO) << "[Controller] Inference workload done";
}

} // namespace ctrl
} // namespace colserve
