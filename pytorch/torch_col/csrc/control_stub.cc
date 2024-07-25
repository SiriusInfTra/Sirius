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

StubBase::StubBase() {
  if (TorchColConfig::HasColocatedInferServer()) {
    CHECK(ctrl::InfTraCommunicator::IsInitialized());
    thread_.reset(new std::thread([this]() {
      running_ = true;
      while (running_) {
        ctrl::CtrlMsgEntry msg;
        bool succ = ctrl::InfTraCommunicator::GetMQ()
            ->TimedGet(1000,
                      ctrl::InfTraMessageQueue::Direction::kInf2Tra,
                      TorchColConfig::GetTrainRank(), msg);
        if (succ) {
          ProcessCtrlMsg(TorchColConfig::GetTrainRank(), msg);
        }
      }
      LOG(INFO) << "[Ctrl Stub] control thread exit";
    }));
  }
}

void StubBase::Stop() {
  if (TorchColConfig::HasColocatedInferServer()) {
    running_ = false;
    thread_->join();
  }
}

int StubBase::GetCmd() {
  return cmd_;
}

void StubBase::SetCmd(int cmd) {
  std::unique_lock lock{mutex_};
  cmd_ = cmd;
}

void StubBase::TrainStart() {
  if (TorchColConfig::HasColocatedInferServer()
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void StubBase::TrainEnd() {
  if (TorchColConfig::HasColocatedInferServer()
      && TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void StubBase::ReportBatchSize(int batch_size) {
  if (TorchColConfig::HasColocatedInferServer()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put(ctrl::CtrlMsgEntry{
                .id = 0,
                .event = static_cast<int>(ctrl::CtrlEvent::kReportBatchSize), 
                .value = batch_size
              },
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void StubBase::StepsNoInteruptBegin() {
  step_mutex_.lock();
  exec_step_ = true;
}

void StubBase::StepsNoInteruptEnd() {
  exec_step_ = false;
  step_mutex_.unlock();
}

bool StubBase::CanExitAfterInferWorkloadDone() {
  constexpr long wait_mill = 30 * 1000; // 30s

  long exit = torch_col::get_unix_timestamp() 
              - infer_workload_done_timestamp_ > wait_mill;

  DLOG(INFO) << "[Check InferWorkload Done]: "
            << " infer_workload_done_timestamp: " 
            << infer_workload_done_timestamp_
            << " current: " << torch_col::get_unix_timestamp()
            << " diff: " 
            << (torch_col::get_unix_timestamp() 
                - infer_workload_done_timestamp_)
            << " exit: " << exit;
  if (infer_workload_done_timestamp_ == 0) {
    return false;
  } else {
    return exit; // 30s
  }
}

void StubBase::EnableTorchColEngine() {
  torch_col::SetUpTorchColEngine(this);
}

void DummyStub::ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) {
  std::unique_lock lock{mutex_};
  switch (msg.event) {
  case static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone):
    infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
    LOG(INFO) << "[DummyStub] inference workload done";
    break;
  default:
    LOG(FATAL) << "[DummyStub] Unknown command: " << msg.event;
  }
}

void SwitchStub::ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) {
  std::unique_lock lock{mutex_};
  switch (msg.event) {
  case (static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)):
    if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) 
        && cmd_id_ == 0) {
      // already interrupting train
      cmd_id_ = msg.id;
      last_reply_cmd_id_ = cmd_id_;
      ctrl::InfTraCommunicator::GetMQ()
          ->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)},
                ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                TorchColConfig::GetTrainRank());
      cmd_id_ = 0;
      LOG(INFO) << "[SwitchStub] already interrupting train, done";
    } else {
      cmd_id_ = msg.id;
      cmd_ = static_cast<int>(ctrl::CtrlEvent::kInterruptTrain);
      LOG(INFO) << "[SwitchStub] Interrupt train";
    }
    break;
  case (static_cast<int>(ctrl::CtrlEvent::kResumeTrain)):
    if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) 
        && cmd_id_ == 0) {
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
  case (static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)):
    this->infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
    LOG(INFO) << "[SwitchStub] Inference workload done";
    break;
  default:
    LOG(FATAL) << "[SwitchStub] Unknown command: " << msg.event;
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

int ColocateStub::GetTargetBatchSize() {
  std::unique_lock locker{mutex_};
  return target_bs_;
}

void ColocateStub::ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) {
  std::unique_lock locker{mutex_};
  switch (static_cast<ctrl::CtrlEvent>(msg.event)) {
  case ctrl::CtrlEvent::kColocateAdjustL1:
  case ctrl::CtrlEvent::kColocateAdjustL2:
    LOG(INFO) << "[ColocateStub] Adjust batch size, target " << msg.value
              << " cur target " << this->target_bs_
              << " current " << this->current_bs_
              << " timestamp: " << torch_col::get_unix_timestamp();
              // << " malloc_ms " << colserve::sta::CUDAMemPool::TrainAllocMs();
    // CHECK_LT(this->target_bs_, this->current_bs_);
    if (msg.value >= this->target_bs_) {
      LOG(INFO) << "[ColocateStub] skip satisfied adjust, reply adjust immediately";
      ctrl::InfTraCommunicator::GetMQ()
          ->Put(ctrl::CtrlMsgEntry{
                  .id = msg.id,
                  .event = msg.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1) 
                           ? static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1Done)
                           : static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2Done)
                },
                ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                TorchColConfig::GetTrainRank());
      break;
    }
    this->target_bs_ = msg.value;

    // only used for colocate l1
    if (TorchColConfig::kill_batch_on_recv 
        && msg.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)) {
        std::unique_lock step_lock{step_mutex_};
        auto t1 = torch_col::get_unix_timestamp();
        sta::xsched::SetRejectCudaCalls(true);
        size_t remove = sta::xsched::AbortStream();
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(
            sta::xsched::GetRegisteredGlobalStream());
        CHECK(reinterpret_cast<uint64_t>(stream) != 0);
        auto err = cudaStreamSynchronize(stream);
        CHECK_EQ(err, cudaSuccess) << "cudaStreamSynchronize failed: " 
                                        << cudaGetErrorString(err);
        auto t2 = torch_col::get_unix_timestamp();
        LOG(INFO) << "Receive adjust request, cancel calls first, "
                  << " cost " << t2 - t1 << "ms, remove " << remove 
                  << " cuda command(s).";
    }
    cmd_ = msg.event;
    cmd_id_ = msg.id;
    set_cmd_time_ = std::chrono::steady_clock::now();
    StubProfiler::RecordAdjustRequest();
    break;
  case ctrl::CtrlEvent::kInferExit: {
    auto old_target_bs = this->target_bs_;
    this->target_bs_ = msg.value;
    LOG(INFO) << "[ColocateStub] Infer Exit adjust, cmd_id " << msg.id
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
    }
    break;
  case ctrl::CtrlEvent::kInferenceWorkloadDone:
    this->infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
    LOG(INFO) << "[ColocateStub] Inference workload done, cmd_id " << msg.id;
    break;
  default:
    LOG(FATAL) << "[ColocateStub] Unknown command: " << msg.event;
  }
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
    LOG(INFO) << "[ColocateStub] Adjust L1 done, timestamp: " 
              << torch_col::get_unix_timestamp();
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
    LOG(INFO) << "[ColocateStub] Adjust L2 done, timestamp: " 
              << torch_col::get_unix_timestamp();
  }
}

double ColocateStub::PassedTimeFromSetCmd() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - set_cmd_time_).count();
}

void ColocateStub::ReportBatchSize(int batch_size) {
  current_bs_ = batch_size;
  StubBase::ReportBatchSize(batch_size);
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