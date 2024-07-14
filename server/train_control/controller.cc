#include <server/logging_as_glog.h>
#include <server/train_launcher.h>
#include <server/resource_manager.h>
#include <server/model_store/infer_model_store.h>
#include <server/model_store/model_cache.h>
#include <server/train_control/controller.h>
#include <server/config.h>

#include <common/util.h>

#include <signal.h>
#include <algorithm>


namespace colserve
{
  
std::unique_ptr<Controller> Controller::controller_;

std::atomic<uint64_t> Controller::adjust_cmd_id = 1;

std::ostream& operator<<(std::ostream&os, ctrl::CtrlEvent event) {
  switch (event) {
  // status event
  case ctrl::CtrlEvent::kTrainStart:
    os << "ctrl::CtrlEvent::kTrainStart"; 
    return os;
  case ctrl::CtrlEvent::kTrainEnd:
    os << "ctrl::CtrlEvent::kTrainEnd"; 
    return os;
  case ctrl::CtrlEvent::kInterruptTrainDone:
    os << "ctrl::CtrlEvent::kInterruptTrainDone";
    return os;
  case ctrl::CtrlEvent::kResumeTrainDone:
    os << "ctrl::CtrlEvent::kResumeTrainDone";
    return os;

  // cmd event
  case ctrl::CtrlEvent::kInterruptTrain:
    os << "ctrl::CtrlEvent::kInterruptTrain";
    return os;
  case ctrl::CtrlEvent::kResumeTrain:
    os << "ctrl::CtrlEvent::kResumeTrain";
    return os;
  default:
    LOG(FATAL) << "unknown ctrl::CtrlEvent";
    return os;
  }
}

std::ostream& operator<<(std::ostream& os, const Controller::TrainStatus &status) {
  switch (status.status) {
  case Controller::TrainStatus::kIdle:
    os << "Controller::TrainStatus::kIdle";
    return os;
  case Controller::TrainStatus::kRunning:
    os << "Controller::TrainStatus::kRunning";
    return os;
  case Controller::TrainStatus::kInterrupted:
    os << "Controller::TrainStatus::kInterrupted";
    return os;
  default:
    LOG(FATAL) << "unknown Controller::TrainStatus";
    return os;
  }
}

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
  // train_cmd_mqs_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("cmd-ctrl", , true);
  // train_status_mqs_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("status-ctrl", true);
  for (int i = 0; i < sta::DeviceManager::GetNumVisibleGpu(); i++) {
    train_cmd_mqs_.push_back(std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>(
        "cmd-ctrl", i, true));
    train_status_mqs_.push_back(std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>(
        "status-ctrl", i, true));
  }
  monitor_train_thread_ = std::make_unique<std::thread>(&Controller::MonitorTrain, this);
}

void Controller::MonitorTrain() {
  while (true) {
    // auto entry = static_cast<ctrl::CtrlMsgEntry>(train_status_mqs_->BlockGet());
    // auto entry = train_status_mqs_[0]->BlockGet();
    ctrl::CtrlMsgEntry entry;
    while (true) {
      bool succ = false;
      for (auto & status_mq : train_status_mqs_) {
        succ = status_mq->TryGet(entry);
        if (succ) break;
      }
      if (succ) break;
    }
    auto event = static_cast<ctrl::CtrlEvent>(entry.event);
    auto cur_status = train_status_;
    // LOG(INFO) << "[Controller] MonitorTrain: " << event;
    switch (event){
    case ctrl::CtrlEvent::kTrainStart:
      CHECK_EQ(train_status_.status, TrainStatus::kIdle);
      train_status_.status = TrainStatus::kRunning;
      break;
    case ctrl::CtrlEvent::kInterruptTrainDone:
      train_status_.status = TrainStatus::kInterrupted;
      LOG(INFO) << "[Controller]: train interrupted";
      break;
    case ctrl::CtrlEvent::kResumeTrainDone:
      CHECK(train_status_.status == TrainStatus::kInterrupted 
          || train_status_.status == TrainStatus::kRunning);
      train_status_.status = TrainStatus::kRunning;
      break;
    case ctrl::CtrlEvent::kColocateAdjustL1Done:
    case ctrl::CtrlEvent::kColocateAdjustL2Done:
      adjust_done_id_ = entry.id;
      wait_train_adjust_cv_.notify_all();
      break;
    case ctrl::CtrlEvent::kTrainEnd:
      CHECK(train_status_.status == TrainStatus::kRunning
          || train_status_.status == TrainStatus::kIdle);
      train_status_.status = TrainStatus::kIdle;
      // train_cmd_mqs_->Clear();
      for (auto & cmd_mq : train_cmd_mqs_) {
        cmd_mq->Clear();
      }
      break;
    case ctrl::CtrlEvent::kReportBatchSize:
      CHECK(train_status_.status != TrainStatus::kIdle);
      TrainLauncher::Get()->SetCurBatchSize(entry.value);
      break;
    default:
      break;
    }
    if (cur_status.status == TrainStatus::kRunning 
        && train_status_.status != TrainStatus::kRunning) {
      wait_train_cv_.notify_all();
    }
    if (cur_status.status != train_status_.status) {
      LOG(INFO) << "[Controller] MonitorTrain: train status " 
                << cur_status << " -> " << train_status_;
    }
  }
}

uint64_t Controller::InterruptTrain() {
  static std::atomic<uint64_t> interrupt_cmd_id = 1;
  auto cmd_id = interrupt_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (Config::serve_mode == ServeMode::kTaskSwitchL1) {
    if (train_status_.status == TrainStatus::kRunning) {
      // LOG(INFO) << "Controller: Put InterruptTrain";
      // train_cmd_mqs_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)});
      for (auto & cmd_mq : train_cmd_mqs_) {
        cmd_mq->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)});
      }
    }
  } else if (Config::serve_mode == ServeMode::kTaskSwitchL3) {
    if (TrainLauncher::Get()->GetTrainPid() != -1) {
      LOG(INFO) << "[Controller]: kill train";
      CHECK_EQ((kill(TrainLauncher::Get()->GetTrainPid(), SIGKILL)), 0);
      // TrainEnd();
    }
  }
  return cmd_id;
}

uint64_t Controller::ResumeTrain() {
  static std::atomic<uint64_t> resume_cmd_id = 1;
  auto cmd_id = resume_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    // LOG(INFO) << "Controller: Put ResumeTrain";
    for (auto & cmd_mq : train_cmd_mqs_) {
      train_cmd_mqs_[0]->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kResumeTrain)});
    }
  }
  return cmd_id;
}

uint64_t Controller::ColocateAdjust(size_t model_rank, int device_id, size_t batch_size) {
  static std::mutex adjust_batch_mutex; // for quick fix concurrency

  auto cmd_id = Controller::adjust_cmd_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
#if ADJUST_WITH_FLYING
    std::lock_guard lock{adjust_batch_mutex};
#endif
    TrainLauncher::Get()->AddTargetBatchSize(-batch_size);
    if (Config::serve_mode == ServeMode::kColocateL1) {
      train_cmd_mqs_[device_id]->Put({
          cmd_id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1), 
          static_cast<int>(TrainLauncher::Get()->GetTargetBatchSize())});
    } else if (Config::serve_mode == ServeMode::kColocateL2) {
      train_cmd_mqs_[device_id]->Put({
        cmd_id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2), 
        static_cast<int>(TrainLauncher::Get()->GetTargetBatchSize())});
    }
  }
  LOG(INFO) << "[Controller] model " << model_rank 
            << " send ColocateAdjust cmd_id: " << cmd_id 
            << " batch_size: " << batch_size
            << " train idle " << IsTrainIdle();
  return cmd_id;
}

uint64_t Controller::InferExit(int device_id) {
  static std::atomic<uint64_t> infer_exit_id = 1;

  auto cmd_id = infer_exit_id.fetch_add(1, std::memory_order_relaxed);
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
        ResourceManager::InferMemoryChangingLock();
        // size_t reserve_memory_MB = sta::ByteToMB(Config::cold_cache_max_capability_nbytes - ColdModelCache::Get().GetCachedNbytes(cold_cache_lock));
        // train_avail_memory_MB = ResourceManager::GetTrainAvailMemoryMB() - reserve_memory_MB;
        // train_avail_memory_MB = std::max(train_avail_memory_MB, 0.0);
        // train_target_bs = TrainLauncher::Get()->PredictTargetBatchSize(train_avail_memory_MB);

        // min of bs predicted based on the actual and the predict
        train_avail_memory_MB = std::max(ResourceManager::GetTrainAvailMemoryMB(false), 0.0);
        free_memory_MB = std::max(ResourceManager::GetFreeMemoryMB(true), 0.0);
        reserve_memory_MB = ColdModelCache::Get(0)->GetReleaseReserveMemoryMB(cold_cache_lock);
        if (Config::use_shared_tensor) {
          auto adjust_bs = TrainLauncher::Get()->GetAdjustBatchSize(
              std::max(free_memory_MB - reserve_memory_MB, 0.0));
          train_bs_predict_by_avail_memory = TrainLauncher::Get()->PredictTargetBatchSize(
              std::max(train_avail_memory_MB - reserve_memory_MB, 0.0));
          cur_train_target_bs = TrainLauncher::Get()->GetTargetBatchSize();
          train_target_bs = std::max(
              cur_train_target_bs, 
              std::min(cur_train_target_bs + adjust_bs, train_bs_predict_by_avail_memory));
        } else {
          // not use availd memory, resulting in wrong prediction
          train_bs_predict_by_avail_memory = -1;
          auto adjust_bs = TrainLauncher::Get()->GetAdjustBatchSize(std::max(free_memory_MB, 0.0));
          cur_train_target_bs = TrainLauncher::Get()->GetTargetBatchSize();
          train_target_bs = std::max(cur_train_target_bs, cur_train_target_bs + adjust_bs);
        }

        TrainLauncher::Get()->SetTargetBatchSize(train_target_bs);
        ResourceManager::InferMemoryChangingUnlock();
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
      train_cmd_mqs_[device_id]->Put({
          cmd_id, static_cast<int>(ctrl::CtrlEvent::kInferExit), 
          train_target_bs});
    }
  }
  return cmd_id;
}

uint64_t Controller::DummyInferExit(int device_id, int target_batch_size) {
  CHECK(Config::dummy_adjust) << "only used for dummy adjust";

  static std::atomic<uint64_t> dummy_infer_exit_id = 1;
  auto cmd_id = dummy_infer_exit_id.fetch_add(1, std::memory_order_relaxed);
  if (!IsTrainIdle()) {
    train_cmd_mqs_[device_id]->Put({
        cmd_id, static_cast<int>(ctrl::CtrlEvent::kInferExit), 
        target_batch_size});
  }
  return cmd_id;
}

bool Controller::WaitTrainNotRunning() {
  std::unique_lock lock{wait_train_mutex_};
  wait_train_cv_.wait(lock, [&]() { return train_status_.status != TrainStatus::kRunning; });
  return true;
}

bool Controller::WaitInferIdle() {
  std::unique_lock lock{wait_infer_mutex_};
  wait_infer_cv_.wait(lock, [&](){ return infer_status_.status == InferStatus::kIdle; });
  return true;
}

bool Controller::WaitColocateAdjustDone(uint64_t cmd_id) {
  if (!IsTrainIdle()) {
    if (Config::IsColocateMode()) {
      std::unique_lock lock{wait_train_adjust_mutex_};
      wait_train_adjust_cv_.wait(lock, [&](){ return adjust_done_id_ >= cmd_id; });
      // auto event = train_adjust_event_mq_->BlockGet();
    }
  }
  return true;
}

void Controller::InferRequestInc(size_t inc) {
  std::unique_lock lock{infer_status_.mutex};
  infer_status_.num_requests += inc;
  infer_status_.status = InferStatus::kRunning;
  // LOG(INFO) << "num_req: " <<infer_status_.num_requests 
  //           << " num_resp: " << infer_status_.num_responses;
}

void Controller::InferResponseInc(size_t inc) {
  std::unique_lock lock{infer_status_.mutex};
  infer_status_.num_responses += inc;
  if (infer_status_.num_responses == infer_status_.num_requests) {
    infer_status_.status = InferStatus::kIdle;
    wait_infer_cv_.notify_one();
    DLOG(INFO) << "[Controller] InferStatus -> kIdle";
  }
  // LOG(INFO) << "num_resp: " << infer_status_.num_responses
  //           << " num_req: " << infer_status_.num_requests;
}

bool Controller::IsInferIdle() {
  std::unique_lock lock{infer_status_.mutex};
  return infer_status_.status == InferStatus::kIdle;
}

void Controller::LogInferStatus() {
  std::unique_lock lock{infer_status_.mutex};
  LOG(INFO) << "[Controller] InferStatus: " << infer_status_.status
            << " num_req: " << infer_status_.num_requests
            << " num_resp: " << infer_status_.num_responses;
}

std::string Controller::GetInferStatusStr() {
  std::stringstream ss;
  std::unique_lock lock{infer_status_.mutex};
  ss << "InferStatus: " << infer_status_.status
     << " num_req: " << infer_status_.num_requests
     << " num_resp: " << infer_status_.num_responses;
  return ss.str();
}

void Controller::TrainStart() {
  // LOG(INFO) << "Controller Put TrainStart";
  train_status_mqs_[0]->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)});
}

void Controller::TrainEnd() {
  train_status_mqs_[0]->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)});
}

bool Controller::IsTrainIdle() {
  return train_status_.status == TrainStatus::kIdle;
}

bool Controller::HasFlyingColocateAdjust() {
  return adjust_done_id_ + 1 < Controller::adjust_cmd_id;
}

void Controller::InferenceWorkloadDone() {
  static std::atomic<uint64_t> infer_workload_done_id = 1;

  auto cmd_id = infer_workload_done_id.fetch_add(1, std::memory_order_relaxed);
  for (int i = 0; i < sta::DeviceManager::GetNumVisibleGpu(); i++) {
    train_cmd_mqs_[i]->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)});
  }
  // train_cmd_mqs_->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)});

  LOG(INFO) << "[Controller] Inference workload done";
}

} // namespace colserve
