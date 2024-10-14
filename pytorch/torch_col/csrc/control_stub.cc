#include <torch_col/csrc/control_stub.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/util.h>
#include <torch_col/csrc/fake_engine.h>
#include <torch_col/csrc/dist_ext.h>
#include <torch_col/csrc/dynamic_batch.h>
#include <torch_col/csrc/dist_train_sync.h>

#include <common/log_as_glog_sta.h>
#include <common/cuda_allocator.h>
#include <common/xsched_ctrl.h>
#include <common/inf_tra_comm/communicator.h>

#include <cstddef>

// [Note: fast training memory adjust]
// fast training memory adjust contains multiple stages:
// 1. Receive adjust request from infer           ┬       ┬      ┬ 
//            |                                   │       │      │ 
//            |                                   │       │      │ 
// 2. clear launched cuda kernels                 │       │      │ 
//    (including nccl for DDP)                    │       │      │ 
//            |                          cmd=AdjustL1     │      │
//            |                                   │       │      │ 
// 3. release memory, then reply to inference     ┴    killed_batch_recover=False
//            |                                   ┬       │      │
//            |                                   │       │      │
//            |                          direct_reply     │      │
//            |                          skip_kill_batch  │      │
//            |                                   │       │      │
// 4. [restart nccl (for DDP)]                    ┴       ┴      │
//            |                                              will_killed_batch_reconfig=True
//            |                                                  │
// 5. reconfigure training                                       ┴
//    (i.e., distributing batch) 
// 6. continue training


// [Note: task switch] (multi-gpu)
// Generally, server will maintain training status to guarantee the validity of
// interrupt/resume training. However, the trick is that the training require
// time to change its status. 
// - Case 1 (interrupt training), as long as resume msg is sent to training, 
//   we choose to let interrupt training to be valid (avoiding starting infer 
//   while training resumes). so, training should handle the interrupt for two 
//   cases, normal running and during resume.
// - Case 2 (resuem training, only send to master), it need to be guaranteed 
//   that all training workers consent to resume. Remind that we need to handle
//   the interrupt before resuming is done. so, we need to sync workers to check
//   whether the all training agree to resume.

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
      LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank()
                <<  " | CtrlStub] control thread exit";
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

void StubBase::SetTrainFirstEpochDone() {
  if (TorchColConfig::HasColocatedInferServer()) {
    COMMUNICATOR_UPDATE_SHARED_TRAIN_INFO_FIELD(
        TorchColConfig::GetTrainRank(), first_epoch_done, true);
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
    LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank()
              << " | DummyStub] inference workload done";
    break;
  default:
    LOG(FATAL) << "[Rank " << TorchColConfig::GetTrainRank() 
               << " | DummyStub] Unknown command: " << msg.event;
  }
}

SwitchStub::SwitchStub() : StubBase() {
  DLOG(INFO) << "[SwitchStub] initialized";

  DistTrainSync::CreateCustomSharedData(
    "switch_stub", 

    std::make_pair(std::string{"global_interrupt_flag"}, 
                   &shared_data_.interrupt_flag_),
    std::make_pair(std::string{"has_batch_killed"}, 
                   &shared_data_.has_batch_killed_),
    std::make_pair(std::string{"mut"}, 
                   &shared_data_.mut_)
  );
};

void SwitchStub::ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) {
  std::unique_lock lock{mutex_};
  switch (msg.event) {
  case (static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)):
    if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) 
        && cmd_id_ == 0) {
      // already interrupting train
      cmd_id_ = msg.id;
      last_reply_cmd_id_ = cmd_id_;
      if (TorchColConfig::IsTrainMaster()) {
        CHECK(GetGlobalHasBatchKilled());
        CHECK(GetGlobalInterruptFlag());
        ctrl::InfTraCommunicator::GetMQ()
            ->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)},
                  ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                  TorchColConfig::GetTrainRank());
        LOG_IF(INFO, TorchColConfig::log_control_stub) 
            << "[Rank "<< TorchColConfig::GetTrainRank()
            << " | SwitchStub] already interrupting train, done, cmd_id: "
            << msg.id;
      }
      cmd_id_ = 0;
    } else {
      if (TorchColConfig::IsTrainMaster()) {
        SetGlobalInterruptFlag(true);
        SetGlobalHasBatchKilled(false);
        LOG_IF(INFO, TorchColConfig::log_control_stub) 
            << "[Rank " << TorchColConfig::GetTrainRank()
            << " | SwitchStub] Interrupt train, cmd_id: " << msg.id;
      }

      if (TorchColConfig::kill_batch_on_recv 
          && has_killed_batch_recover_) {
        CHECK(cmd_ == static_cast<int>(ctrl::CtrlEvent::kResumeTrain) 
              || cmd_ == -1 /* the first time */) << "cmd_ " << cmd_;
        std::unique_lock step_lock{step_mutex_};
        auto t1 = torch_col::get_unix_timestamp();
        // DLOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank()
        //           << " | SwitchStub] kill batch, cmd " << cmd_
        //           << " cmd_id " << cmd_id_;
        sta::xsched::SetRejectCudaCalls(true);
        if (ProcessGroupNCCL::HasDefaultProcessGroupNCCL()) {
          ProcessGroupNCCL::GetDefaultProcessGroupNCCL()->SetNcclCommAbortFlag(
            {at::Device(at::kCUDA, TorchColConfig::GetTrainRank())});
        }
        auto remove = sta::xsched::AbortAllStreams();
        sta::xsched::SyncAllStreams();
        auto t2 = torch_col::get_unix_timestamp();
        has_killed_batch_recover_ = false;
        LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank()
                  << " | SwitchStub] kill batch, remove " << remove
                  << ", cost " << t2 - t1 << "ms";
      }
      cmd_id_ = msg.id;
      cmd_ = static_cast<int>(ctrl::CtrlEvent::kInterruptTrain);
    }
    LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank()
              << " | SwitchStub] Interrupt train, cmd " << cmd_
              << " cmd_id: " << cmd_id_;
    break;
  case (static_cast<int>(ctrl::CtrlEvent::kResumeTrain)):
    if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) 
        && cmd_id_ == 0) {
      // already interrupting train
      cmd_ = static_cast<int>(ctrl::CtrlEvent::kResumeTrain);
      SetGlobalInterruptFlag(false);
      LOG_IF(INFO, TorchColConfig::log_control_stub) 
          << "[Rank " << TorchColConfig::GetTrainRank() 
          << " | SwitchStub] Resume train, cmd " << cmd_;

      // if (TorchColConfig::IsTrainMaster()) {
      //   ctrl::InfTraCommunicator::GetMQ()
      //       ->Put({0, static_cast<int>(ctrl::CtrlEvent::kResumeTrainDone)},
      //             ctrl::InfTraMessageQueue::Direction::kTra2Inf,
      //             TorchColConfig::GetTrainRank());
      // }
    } else {
      // should not happen if cmd_ is kInterruptTrain (ie. 7)
      LOG_IF(INFO, TorchColConfig::log_control_stub) 
          << "[Rank " << TorchColConfig::GetTrainRank()
          << "| SwitchStub] Ignore resume train, cmd " << cmd_;
    }
    break;
  case (static_cast<int>(ctrl::CtrlEvent::kInferenceWorkloadDone)):
    this->infer_workload_done_timestamp_ = torch_col::get_unix_timestamp();
    LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank()
              << " | SwitchStub] Inference workload done";
    break;
  default:
    LOG(FATAL) << "[Rank "<< TorchColConfig::GetTrainRank()
               << " | SwitchStub] Unknown command: " << msg.event;
  }
}

bool SwitchStub::TryInterruptTrainDone(bool barrier) {
  std::unique_lock locker{mutex_};
  return TryInterruptTrainDoneWithLock(barrier);
}

bool SwitchStub::TryInterruptTrainDoneWithLock(bool barrier) {
  if (cmd_id_ > last_reply_cmd_id_) {
    auto cmd_id = cmd_id_;
    last_reply_cmd_id_ = cmd_id_;
    cmd_id_ = 0;
    sta::xsched::SetRejectCudaCalls(false); // should move to `SetKillBatchRecover` ?
    if (barrier) {
      DistTrainSync::WaitBarrier();
    }
    if (TorchColConfig::IsTrainMaster()) {
      ctrl::InfTraCommunicator::GetMQ()
          ->Put({cmd_id, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)},
                ctrl::InfTraMessageQueue::Direction::kTra2Inf,
                TorchColConfig::GetTrainRank());
      LOG_IF(INFO, TorchColConfig::log_control_stub) 
          << "[Rank " << TorchColConfig::GetTrainRank()
          << " | SwitchStub] Interrupt train done, cmd_id: " 
          << last_reply_cmd_id_;
    }
    return true;
  }
  return false;
}

void SwitchStub::TrainResumeDone() {
  if (TorchColConfig::IsTrainMaster()) {
    std::unique_lock locker{mutex_};
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kResumeTrainDone)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }
}

void SwitchStub::SetKilledBatchRecover() {
  has_killed_batch_recover_.store(true, std::memory_order_release);
  DLOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
            << " | SwitchStub] Set killed batch recover";
}

void SwitchStub::SetGlobalInterruptFlag(bool flag) {
  bip::scoped_lock lock{*shared_data_.mut_};
  *shared_data_.interrupt_flag_ = flag;
}

// call by master
bool SwitchStub::PrepareResume() {
  std::unique_lock lock{mutex_};
  
  bool resume = true;
  {
    if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)
        && cmd_id_ != 0) {
      LOG_IF(INFO, TorchColConfig::log_control_stub) 
          << "[Rank " << TorchColConfig::GetTrainRank() 
          << " | SwitchStub] Prepare to resume train, "
          << "find interrupt, cmd_id: " << cmd_id_;
      SetGlobalInterruptFlag(true);
    }
    DistTrainSync::WaitBarrier();
    resume &= !GetGlobalInterruptFlag();
    DistTrainSync::WaitBarrier();
  }

  if (!resume) {
    TryInterruptTrainDoneWithLock(false);
    DistTrainSync::WaitBarrier();
    return false;
  }

  CHECK(!TorchColConfig::IsTrainMaster() 
      || cmd_ == static_cast<int>(ctrl::CtrlEvent::kResumeTrain));

  cmd_ = static_cast<int>(ctrl::CtrlEvent::kResumeTrain);
  has_killed_batch_recover_ = true; // control whether batch can be killed
  DistTrainSync::WaitBarrier();

  if (TorchColConfig::IsTrainMaster()) {
    ctrl::InfTraCommunicator::GetMQ()
        ->Put({0, static_cast<int>(ctrl::CtrlEvent::kResumeTrainDone)},
              ctrl::InfTraMessageQueue::Direction::kTra2Inf,
              TorchColConfig::GetTrainRank());
  }

  LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
            << " | SwitchStub] Prepare to resume train,"
            << " cmd " << cmd_ << " cmd_id: " << cmd_id_
            << " has_killed_batch_recover " << has_killed_batch_recover_
            << " global_interrupt_flag " << *shared_data_.interrupt_flag_
            << (ProcessGroupNCCL::HasDefaultProcessGroupNCCL() 
                ? str(boost::format(" nccl_abort_flag %d") 
                      % ProcessGroupNCCL::GetDefaultProcessGroupNCCL()->GetAbortFlag())
                : "");

  return true;
}

bool SwitchStub::GetGlobalInterruptFlag() {
  bip::scoped_lock lock{*shared_data_.mut_};
  return *shared_data_.interrupt_flag_;
}

int ColocateStub::GetTargetBatchSize() {
  std::unique_lock locker{mutex_};
  // return target_bs_;
  if (TorchColConfig::HasColocatedInferServer()) {
    return COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
        TorchColConfig::GetTrainRank(), target_batch_size);
  } else {
    return input_batch_size_;
  }
}

int ColocateStub::GetUnpubTargetBatchSize() {
  std::unique_lock locker{mutex_};
  if (TorchColConfig::HasColocatedInferServer()) {
    return COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
        TorchColConfig::GetTrainRank(), target_batch_size_unpublished);
  } else {
    return input_batch_size_;
  }
}

void ColocateStub::ProcessCtrlMsg(int id, const ctrl::CtrlMsgEntry &msg) {
  std::unique_lock locker{mutex_};
  switch (static_cast<ctrl::CtrlEvent>(msg.event)) {
  case ctrl::CtrlEvent::kColocateAdjustL1:
  case ctrl::CtrlEvent::kColocateAdjustL2: 
  {
//     auto cur_target_bs = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
//         TorchColConfig::GetTrainRank(), target_batch_size);
//     auto unpub_target_bs = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
//         TorchColConfig::GetTrainRank(), target_batch_size_unpublished);
//     LOG_IF(INFO, TorchColConfig::log_control_stub) 
//         << "[Rank " << TorchColConfig::GetTrainRank() <<  " | ColocateStub]" 
//         << " Adjust batch size"
//         << " cur_target_bs " << cur_target_bs
//         << " unpub_target_bs " << unpub_target_bs
//         << " current " << this->current_bs_
//         << " timestamp: " << torch_col::get_unix_timestamp();

#define COLOCATE_ADJUST_IMMEDIATE_REPLY do { \
    ctrl::InfTraCommunicator::GetMQ() \
        ->Put(ctrl::CtrlMsgEntry{ \
            .id = msg.id, \
            .event = msg.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1) \
                     ? static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1Done) \
                     : static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2Done) \
          }, \
          ctrl::InfTraMessageQueue::Direction::kTra2Inf, \
          TorchColConfig::GetTrainRank()); \
 } while (0);

//     if (msg.value >= cur_target_bs
//         || msg.value >= current_bs_) {
//       LOG_IF(INFO, TorchColConfig::log_control_stub) 
//           << "[Rank " << TorchColConfig::GetTrainRank()
//           << " | ColocateStub] skip satisfied adjust, "
//           << "reply adjust immediately";

//       COLOCATE_ADJUST_IMMEDIATE_REPLY
//       break;
//     }

    // only used for colocate l1
    bool should_kill_batch = TorchColConfig::kill_batch_on_recv 
        && msg.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)
        && cmd_ != static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1);

    // [Note: kill batch] (colocate)
    // avoid set nccl abort flag while re-creating new nccl comm, 
    // which long waiting time to get nccl comm before abort.
    // Ref: [Note: fast training memory adjust]
    if (should_kill_batch 
        && !has_killed_batch_recover_.load(std::memory_order_acquire)) {
      LOG_IF(INFO, TorchColConfig::log_control_stub) 
          << "[Rank " << TorchColConfig::GetTrainRank() 
          << " | ColocateStub]" << " Receive adjust request, "
          << "batch is not recovered yet, reply immediately";
    
      COLOCATE_ADJUST_IMMEDIATE_REPLY
      break;
    }
    has_killed_batch_recover_ = false;
    will_killed_batch_reconfig_ = true;

    if (should_kill_batch) {
        std::unique_lock step_lock{step_mutex_};

        auto t1 = torch_col::get_unix_timestamp();
        sta::xsched::SetRejectCudaCalls(true);
        auto _t11 = torch_col::get_unix_timestamp();
        if (ProcessGroupNCCL::HasDefaultProcessGroupNCCL()) {
          ProcessGroupNCCL::GetDefaultProcessGroupNCCL()->SetNcclCommAbortFlag(
              {at::Device(at::kCUDA, TorchColConfig::GetTrainRank())});
        }
        auto _t12 = torch_col::get_unix_timestamp();
        size_t remove = sta::xsched::AbortAllStreams();
        auto t2 = torch_col::get_unix_timestamp();

        sta::xsched::SyncAllStreams();
        auto t3 = torch_col::get_unix_timestamp();
        LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() << "] " 
                  << "Receive adjust request, cancel calls first,"
                  << " cost " << t3 - t1 << "ms (wait kernel "
                  << t3 - t2 << "ms), remove " << remove 
                  << " cuda command(s)"
                  << " | SetRejectCudaCalls " << _t11 - t1 << "ms"
                  << " SetNcclCommAbortFlag " << _t12 - _t11 << "ms"
                  << " AbortAllStreams " << t2 - _t12 << "ms";
    }
    cmd_ = msg.event;
    cmd_id_ = msg.id;
    set_cmd_time_ = std::chrono::steady_clock::now();
    StubProfiler::RecordAdjustRequest();
  }
    break;
  case ctrl::CtrlEvent::kInferExit: 
  {
    // auto cur_target_bs = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
    //     TorchColConfig::GetTrainRank(), target_batch_size);
    // auto unpub_target_bs = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(
    //     TorchColConfig::GetTrainRank(), target_batch_size_unpublished);

    // auto old_target_bs = this->target_bs_;
    // this->target_bs_ = msg.value;

    if (TorchColConfig::IsTrainMaster()) {
      auto target_bs_vec = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD_VEC(
          target_batch_size);
      auto target_bs_unpub_vec = COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD_VEC(
          target_batch_size_unpublished);

      std::stringstream ss;
      if (TorchColConfig::log_control_stub) {
        ss << "[Rank " << TorchColConfig::GetTrainRank() 
           << " | ColocateStub]" 
           << " Infer Exit adjust, cmd_id " << msg.id
           << " cur_target_bs_vec " << target_bs_vec
           << " unpub_target_bs_vec " << target_bs_unpub_vec
           << " timestamp: " << torch_col::get_unix_timestamp();
      }

      if (will_killed_batch_reconfig_.load(std::memory_order_acquire)) {
        // concurrenctly distributing batch with infer require memory adjust
        // may cause some issues. 
        // For example:
        //   kill a sync batch, followed by a infer release memory adjust,
        //   which triggers distributing batch but counts ongoing 
        //   training workers will be wrong.
        LOG_IF(INFO, TorchColConfig::log_control_stub) 
            << ss.str() 
            << ", killed batch will be reconfiged, do nothing here";
        break;
      }

      bool all_same = true;
      for (auto i : boost::irange(TorchColConfig::GetTrainWorldSize())) {
        if (target_bs_vec[i] != target_bs_unpub_vec[i]) {
          all_same = false;
          break;
        }
      }
      LOG_IF(INFO, TorchColConfig::log_control_stub) 
          << ss.str()
          << (all_same ? ", not need to distribute batch" : "");

      if (!all_same) {
        DynamicBatchDistirbutor::DistributeBatch(true, false, false);
      }
    }

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
    LOG(INFO) << "[Rank " << TorchColConfig::GetTrainRank() 
              << " | ColocateStub]" 
              << " Inference workload done, cmd_id " 
              << msg.id;
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
    LOG_IF(INFO, TorchColConfig::log_control_stub) 
        << "[ColocateStub] Adjust L1 done, timestamp: " 
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
    LOG_IF(INFO, TorchColConfig::log_control_stub) 
        << "[ColocateStub] Adjust L2 done, timestamp: " 
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
  StubProfiler::adjust_request_time_stamp_.push_back(
      torch_col::get_unix_timestamp_us());
}

void StubProfiler::RecordAdjustDone() {
  std::unique_lock lock{StubProfiler::mutex_};
  StubProfiler::adjsut_done_time_stamp_.push_back(
      torch_col::get_unix_timestamp_us());
}

}  // namespace torch_col