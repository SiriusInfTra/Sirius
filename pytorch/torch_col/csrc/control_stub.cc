#include <common/log_as_glog_sta.h>
#include <common/cuda_allocator.h>
#include <common/mempool.h>

#include <torch_col/csrc/control_stub.h>
#include <torch_col/csrc/config.h>
#include <torch_col/csrc/util.h>
#include <torch_col/csrc/fake_engine.h>

#include <PySched.h>
#include <cstddef>


namespace torch_col {

using namespace colserve;

std::vector<long> StubProfiler::adjust_request_time_stamp_;
std::vector<long> StubProfiler::adjsut_done_time_stamp_;
std::mutex StubProfiler::mutex_;

SwitchStub::SwitchStub() {
  cmd_event_mq_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("cmd-ctrl", false);
  status_event_mq_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("status-ctrl", false);

  thread_.reset(new std::thread([&](){
    while (running_) {
      ctrl::CtrlMsgEntry data;
      bool succ = cmd_event_mq_->TimedGet(data, 1000);
      if (succ) {
        std::unique_lock locker{mutex_};
        if (data.event == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain)) {
          if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kInterruptTrain) && cmd_id_ == 0) { // already interrupting train
            cmd_id_ = data.id;
            last_reply_cmd_id_ = cmd_id_;
            status_event_mq_->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)});
            cmd_id_ = 0;
            LOG(INFO) << "[SwitchStub] already interrupting train, done";
          } else {
            cmd_id_ = data.id;
            cmd_ = static_cast<int>(ctrl::CtrlEvent::kInterruptTrain);
            LOG(INFO) << "[SwitchStub] Interrupt train";
          }
          // status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)});
        } else if (data.event == static_cast<int>(ctrl::CtrlEvent::kResumeTrain)) {
          cmd_ = static_cast<int>(ctrl::CtrlEvent::kResumeTrain);
          LOG(INFO) << "[SwitchStub] Resume train";
          status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kResumeTrainDone)});
        } else {
          LOG(FATAL) << "[SwitchStub] Unknown command: " << data.event;
        }
      } else {
        // LOG(INFO) << "[SwitchStub] No command";
      }
    }
  }));
}

void SwitchStub::Stop() {
  running_ = false;
  thread_->join();
}

void SwitchStub::TrainStart() {
  status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)});
}

void SwitchStub::TrainEnd() {
  status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)});
}

bool SwitchStub::TryInterruptTrainDone() {
  std::unique_lock locker{mutex_};
  // LOG(INFO) << "TryInterruptTrainDone " << cmd_id_ << " " << last_reply_cmd_id_; 
  if (cmd_id_ > last_reply_cmd_id_) {
    last_reply_cmd_id_ = cmd_id_;
    status_event_mq_->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kInterruptTrainDone)});
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
  status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kReportBatchSize), batch_size});
}

void SwitchStub::StepsNoInteruptBegin() {
  // std::unique_lock lock{mutex_};/
  exec_step_ = true;
}

void SwitchStub::StepsNoInteruptEnd() {
  // std::unique_lock lock{mutex_};
  exec_step_ = false;
}


ColocateStub::ColocateStub(int batch_size) : target_bs_(batch_size), current_bs_(batch_size) {
  // char *has_server_env = std::getenv("HAS_INFER_SERVER");
  // bool has_server = has_server_env == nullptr ? true : (std::string(has_server_env) == "1");
  cmd_event_mq_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("cmd-ctrl", !has_colocated_infer_server);
  status_event_mq_ = std::make_unique<MemoryQueue<ctrl::CtrlMsgEntry>>("status-ctrl", !has_colocated_infer_server);
  // adjust_event_mq_ = std::make_unique<MemoryQueue<int>>("adjust-ctrl", false);

  thread_.reset(new std::thread([&]() {
    while (running_) {
      ctrl::CtrlMsgEntry data;
      bool succ = cmd_event_mq_->TimedGet(data, 1000);
      if (succ) {
        if (data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)
            || data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2)) {
          std::unique_lock locker{mutex_};
          this->target_bs_ -= data.value;
          LOG(INFO) << "[ColocateStub] Adjust batch size, target " << this->target_bs_ 
                    << " current " << this->current_bs_
                    << " timestamp: " << torch_col::get_unix_timestamp()
                    << " malloc_ms " << colserve::sta::CUDAMemPool::TrainAllocMs();
          // CHECK_LT(this->target_bs_, this->current_bs_);
          if (this->target_bs_ >= this->current_bs_) {
            LOG(INFO) << "[ColocateStub] skip satisfied adjust, reply adjust immediately";
            if (data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)) {
              status_event_mq_->Put({data.id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1Done)});
            } else if (data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2)) {
              status_event_mq_->Put({data.id, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2Done)});
            }
            continue;
          }

          // only used for colocate l1
          if (kill_batch_on_recv && data.event == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1)) {
            // TODO: better cancel kernel launch
            std::unique_lock step_lock{step_mutex_};
            auto t1 = torch_col::get_unix_timestamp();
            SetBlockCudaCalls_v2(true);
            size_t remove = AbortStream();
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(GetRegisteredGlobalStream());
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
          LOG(INFO) << "[ColocateStub] Infer Exit adjust " 
                    << old_target_bs << " -> " << this->target_bs_
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
        } else {
          LOG(FATAL) << "[ColocateStub] Unknown command: " << data.event;
        }
      } else {
        // LOG(INFO) << "[ColocateStub] No command";
      }
    }
  }));
}

void ColocateStub::Stop() {
  running_ = false;
  thread_->join();
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
    status_event_mq_->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL1Done)});
    cmd_ = -1;
    cmd_id_ = 0;
    SetBlockCudaCalls_v2(false);
    // colserve::sta::CUDAMemPool::EnableTrainAlloc();
    StubProfiler::RecordAdjustDone();
    LOG(INFO) << "[ColocateStub] Adjust L1 done, timestamp: " << torch_col::get_unix_timestamp()
              << " train_alloc_ms " << colserve::sta::CUDAMemPool::TrainAllocMs();
  }
}

void ColocateStub::ColocateAdjustL2Done() {
  std::unique_lock locker{mutex_};
  if (cmd_ == static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2)) {
    status_event_mq_->Put({cmd_id_, static_cast<int>(ctrl::CtrlEvent::kColocateAdjustL2Done)});
    cmd_ = -1;
    cmd_id_ = 0;
    StubProfiler::RecordAdjustDone();
    LOG(INFO) << "[ColocateStub] Adjust L2 done, timestamp: " << torch_col::get_unix_timestamp();
  }
}

void ColocateStub::TrainStart() {
  status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainStart)});
}

void ColocateStub::TrainEnd() {
  status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kTrainEnd)});
}

double ColocateStub::PassedTimeFromSetCmd() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(now - set_cmd_time_).count();
}

void ColocateStub::ReportBatchSize(int batch_size) {
  // char *has_server_env = std::getenv("HAS_INFER_SERVER");
  current_bs_ = batch_size;
  // bool has_server = has_server_env == nullptr ? true : (std::string(has_server_env) == "1");
  if (has_colocated_infer_server) {
    status_event_mq_->Put({0, static_cast<int>(ctrl::CtrlEvent::kReportBatchSize), batch_size});
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

void StubBase::EnableTorchColEngine() {
  torch_col::SetUpTorchColEngine(this);
}
}  // namespace torch_col