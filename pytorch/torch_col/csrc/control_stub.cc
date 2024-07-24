#include <common/log_as_glog_sta.h>
#include <common/cuda_allocator.h>
#include <common/xsched_ctrl.h>

#include <torch_col/csrc/control_stub.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/util.h>
#include <torch_col/csrc/fake_engine.h>

#include <cstddef>


namespace torch_col {

using namespace colserve;

std::vector<long> StubProfiler::adjust_request_time_stamp_;
std::vector<long> StubProfiler::adjsut_done_time_stamp_;
std::mutex StubProfiler::mutex_;

bool __CanExitAfterInferWorkloadDone(long infer_workload_done_timestamp) {
  constexpr long wait_mill = 30 * 1000; // 30s
  DLOG(INFO) << "[Check InferWorkload Done]: "
            << " infer_workload_done_timestamp: " << infer_workload_done_timestamp
            << " current: " << torch_col::get_unix_timestamp()
            << " diff: " << torch_col::get_unix_timestamp() - infer_workload_done_timestamp
            << " exit: " << (torch_col::get_unix_timestamp() - infer_workload_done_timestamp > wait_mill);
  if (infer_workload_done_timestamp == 0) {
    return false;
  } else {
    return (torch_col::get_unix_timestamp() - infer_workload_done_timestamp > wait_mill); // 30s
  }
}

StubBase::StubBase() {

}

void StubBase::EnableTorchColEngine() {
  torch_col::SetUpTorchColEngine(this);
}

DummyStub::DummyStub() {
  if (TorchColConfig::has_colocated_infer_server) {
    // ctrl::InfTraCommunicator::Init(false, false, 
    //                                TorchColConfig::GetTrainWorldSize());
    CHECK(ctrl::InfTraCommunicator::IsInitialized());

    thread_.reset(new std::thread([&]() {
      running_ = true;
      while (running_) {
        ctrl::CtrlMsgEntry msg;
        bool succ = ctrl::InfTraCommunicator::GetMQ()
            ->TimedGet(1000, 
                      ctrl::InfTraMessageQueue::Direction::kInf2Tra, 
                      TorchColConfig::GetTrainRank(), msg);
        if (succ) {
          std::unique_lock locker{mutex_};
          if (msg.event == static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)) {
            infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
            LOG(INFO) << "[DummyStub] inference workload done";
          } else {
            LOG(FATAL) << "[DummyStub] Unknown command: " << msg.event;
          }
        }
      }
      LOG(INFO) << "[DummyStub] control thread exit";
    }));
  }
}

void DummyStub::Stop() {
  running_ = false;
  if (TorchColConfig::has_colocated_infer_server) {
    thread_->join();
  }
}

void DummyStub::TrainStart() {
  if (TorchColConfig::has_colocated_infer_server
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void DummyStub::TrainEnd() {
  if (TorchColConfig::has_colocated_infer_server
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)}, 
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

SwitchStub::SwitchStub() {
  if (!TorchColConfig::has_colocated_infer_server) {
    return ;
  }

  // ctrl::InfTraCommunicator::Init(false, false, 
  //                                TorchColConfig::GetTrainWorldSize());
  CHECK(ctrl::InfTraCommunicator::IsInitialized());

  thread_.reset(new std::thread([&](){
    while (running_) {
      ctrl::CtrlMsgEntry data;
      bool succ = ctrl::InfTraCommunicator::GetMQ()->TimedGet(
          1000, ctrl::InfTraMessageQueue::Direction::kInf2Tra, 
          TorchColConfig::GetTrainRank(), data);
      if (succ) {
        std::unique_lock locker{mutex_};
        if (data.event == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)) {
          if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) && cmd_id_ == 0) {
            // already interrupting train
            cmd_id_ = data.id;
            last_reply_cmd_id_ = cmd_id_;
            ctrl::InfTraCommunicator::GetMQ()
                ->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)},
                      ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                      TorchColConfig::GetTrainRank());
            cmd_id_ = 0;
            LOG(INFO) << "[SwitchStub] already interrupting train, done";
          } else {
            cmd_id_ = data.id;
            cmd_ = static_cast<int>(ctrl::CtrlEvent::kInterruptTrain);
            LOG(INFO) << "[SwitchStub] Interrupt train";
          }
          // status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)});
        } else if (data.event == static_cast<int>(ctrl::CtrlEvent::kResumeTrain)) {
          if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) && cmd_id_ == 0) {
            // already interrupting train
            cmd_ = static_cast<int>(ctrl::CtrlEvent::kResumeTrain);
            LOG(INFO) << "[SwitchStub] Resume train";
            ctrl::InfTraCommunicator::GetMQ()
                ->Put({0, static_cast<int>(ctrl::CtrlEvent::kResumeTrainDone)},
                      ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                      TorchColConfig::GetTrainRank());
          } else {
              LOG(INFO) << "[SwitchStub] Ignore resume train";
            }
        } else if (data.event == static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)) {
          this->infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
          LOG(INFO) << "[SwitchStub] Inference workload done";
        } else {
          LOG(FATAL) << "[SwitchStub] Unknown command: " << data.event;
        }
      } else {
        // LOG(INFO) << "[SwitchStub] No command";
      }
    }
    LOG(INFO) << "[SwitchStub] control thread exit";
  }));
}

void SwitchStub::Stop() {
  running_ = false;
  thread_->join();
}

void SwitchStub::TrainStart() {
  if (TorchColConfig::has_colocated_infer_server
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void SwitchStub::TrainEnd() {
  if (TorchColConfig::has_colocated_infer_server
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

bool SwitchStub::TryInterruptTrainDone() {
  std::unique_lock locker{mutex_};
  if (cmd_id_ > last_reply_cmd_id_) {
    last_reply_cmd_id_ = cmd_id_;
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
    cmd_id_ = 0;
    LOG(INFO) << "[SwitchStub] Interrupt train done";
    return true;
  }
  return false;
}

int SwitchStub::Cmd() {
  return cmd_;
}

void SwitchStub::Cmd(int cmd) {
  std::unique_lock locker{mutex_};
  cmd_ = cmd;
}

void SwitchStub::ReportBatchSize(int batch_size) {
  if (TorchColConfig::has_colocated_infer_server) { 
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({
                  .id = 0, 
                  .event = static_cast<int>(ctrl::CtrlEvent::kReportBatchSize), 
                  .value = batch_size
              },
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void SwitchStub::StepsNoInteruptBegin() {
  // std::unique_lock lock{mutex_};/
  exec_step_ = true;
}

void SwitchStub::StepsNoInteruptEnd() {
  // std::unique_lock lock{mutex_};
  exec_step_ = false;
}


ColocateStub::ColocateStub(int batch_size) 
    : target_bs_(batch_size), current_bs_(batch_size) {
  if (!TorchColConfig::has_colocated_infer_server) {
    return ;
  }

  // ctrl::InfTraCommunicator::Init(false, false, 
  //                                TorchColConfig::GetTrainWorldSize());
  CHECK(ctrl::InfTraCommunicator::IsInitialized());

  thread_.reset(new std::thread([&]() {
    while (running_) {
      ctrl::CtrlMsgEntry data;
      bool succ = ctrl::InfTraCommunicator::GetMQ()->TimedGet(
          1000, ctrl::InfTraMessageQueue::Direction::kInf2Tra, 
          TorchColConfig::GetTrainRank(), data);
      if (succ) {
        if (data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)
            || data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2)) {
          std::unique_lock locker{mutex_};
          LOG(INFO) << "[ColocateStub] Adjust batch size, target " << data.value
                    << " cur target " << this->target_bs_
                    << " current " << this->current_bs_
                    << " timestamp: " << torch_col::get_unix_timestamp();
                    // << " malloc_ms " << colserve::sta::CUDAMemPool::TrainAllocMs();
          // CHECK_LT(this->target_bs_, this->current_bs_);
          if (data.value >= this->target_bs_) {
            LOG(INFO) << "[ColocateStub] skip satisfied adjust, reply adjust immediately";
            ctrl::InfTraCommunicator::GetMQ()
                ->Put({.id = data.id,
                       .event = data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1) 
                                ? static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1Done)
                                : static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2Done)
                      },
                      ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                      TorchColConfig::GetTrainRank());
            continue;
          }
          this->target_bs_ = data.value;

          // only used for colocate l1
          if (TorchColConfig::kill_batch_on_recv 
              && data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)) {
            // TODO: better cancel kernel launch
            std::unique_lock step_lock{step_mutex_};
            auto t1 = torch_col::get_unix_timestamp();
            // SetBlockCudaCalls_v2(true);
            // size_t remove = AbortStream();
            colserve::sta::xsched::SetRejectCudaCalls(true);
            size_t remove = colserve::sta::xsched::AbortStream();
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(
                colserve::sta::xsched::GetRegisteredGlobalStream());
            CHECK(reinterpret_cast<uint64_t>(stream) != 0);
            auto err = cudaStreamSynchronize(stream);
            CHECK_EQ(err, cudaSuccess) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
            auto t2 = torch_col::get_unix_timestamp();
            LOG(INFO) << "Receive adjust request, cancel calls first, "
                      << " cost " << t2 - t1 << "ms, remove " << remove << " cuda command(s).";
          }
          cmd_ = data.event;
          cmd_id_ = data.id;
          set_cmd_time_ = std::chrono::steady_clock::now();
          StubProfiler::RecordAdjustRequest();
        } else if (data.event == static_cast<int>(ctrl::CtrlEvent::kInferExit)) {
          auto old_target_bs = this->target_bs_;
          this->target_bs_ = data.value;
          LOG(INFO) << "[ColocateStub] Infer Exit adjust, cmd_id " << data.id
                    << " target bs " << old_target_bs << " -> " << this->target_bs_
                    << " current " << this->current_bs_
                    << " timestamp: " << torch_col::get_unix_timestamp();
          // CHECK_LE(this->target_bs_, this->current_bs_);
          // if (this->target_bs_ == this->current_bs_) {
          //   if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)) {
          //     this->ColocateAdjustL1Done();
          //   } else if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2)) {
          //     this->ColocateAdjustL2Done();
          //   }
          // }
        } else if (data.event == static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)) {
          this->infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
          LOG(INFO) << "[ColocateStub] Inference workload done, cmd_id " << data.id;
        } else {
          LOG(FATAL) << "[ColocateStub] Unknown command: " << data.event;
        }
      } else {
        // LOG(INFO) << "[ColocateStub] No command";
      }
    }
    LOG(INFO) << "[ColocateStub] control thread exit";
  }));
}

void ColocateStub::Stop() {
  if (TorchColConfig::has_colocated_infer_server) {
    running_ = false;
    thread_->join();
  }
}

int ColocateStub::Cmd() {
  std::unique_lock locker{mutex_};
  return cmd_;
}

int ColocateStub::TargetBatchSize() {
  std::unique_lock locker{mutex_};
  return target_bs_;
}

void ColocateStub::ColocateAdjustL1Done() {
  std::unique_lock locker{mutex_};
  if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1Done)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
    cmd_ = -1;
    cmd_id_ = 0;
    // SetBlockCudaCalls_v2(false);
    colserve::sta::xsched::SetRejectCudaCalls(false);
    // colserve::sta::CUDAMemPool::EnableTrainAlloc();
    StubProfiler::RecordAdjustDone();
    LOG(INFO) << "[ColocateStub] Adjust L1 done, timestamp: " << torch_col::get_unix_timestamp();
              // << " train_alloc_ms " << colserve::sta::CUDAMemPool::TrainAllocMs();
  }
}

void ColocateStub::ColocateAdjustL2Done() {
  std::unique_lock locker{mutex_};
  if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2)) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2Done)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
    cmd_ = -1;
    cmd_id_ = 0;
    StubProfiler::RecordAdjustDone();
    LOG(INFO) << "[ColocateStub] Adjust L2 done, timestamp: " << torch_col::get_unix_timestamp();
  }
}

void ColocateStub::TrainStart() {
  if (TorchColConfig::has_colocated_infer_server
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
    LOG(INFO) << "[ColocateStub] Train start";
    // status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)});
  }
}

void ColocateStub::TrainEnd() {
  if (TorchColConfig::IsTrainMaster()) {
    // status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)});
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

double ColocateStub::PassedTimeFromSetCmd() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - set_cmd_time_).count();
}

void ColocateStub::ReportBatchSize(int batch_size) {
  // char *has_server_env = std::getenv("COL_HAS_INFER_SERVER");
  current_bs_ = batch_size;
  // bool has_server = has_server_env == nullptr ? true : (std::string(has_server_env) == "1");
  if (TorchColConfig::has_colocated_infer_server) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({
                  .id = 0, 
                  .event = static_cast<int>(ctrl::CtrlEvent::kReportBatchSize), 
                  .value = batch_size
              },
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
 
}

void ColocateStub::StepsNoInteruptBegin() {
  // std::unique_lock lock{mutex_};
  step_mutex_.lock();
  exec_step_ = true;
}

void ColocateStub::StepsNoInteruptEnd() {
  // std::unique_lock lock{mutex_};
  exec_step_ = false;
  step_mutex_.unlock();
}


void StubProfiler::RecordAdjustRequest() {
  std::unique_lock lock{StubProfiler::mutex_};
  StubProfiler::adjust_request_time_stamp_.push_back(torch_col::get_unix_timestamp_us());
}

void StubProfiler::RecordAdjustDone() {
  std::unique_lock lock{StubProfiler::mutex_};
  StubProfiler::adjsut_done_time_stamp_.push_back(torch_col::get_unix_timestamp_us());
}

}  // namespace torch_col