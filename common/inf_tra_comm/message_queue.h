#pragma once

#include <common/inf_tra_comm/bip_helper.h>
#include <common/util.h>

#include <boost/format.hpp>
#include <array>

namespace colserve {
namespace ctrl {

struct CtrlMsgEntry {
  uint64_t id;
  int event;
  int value;
};

enum class CtrlEvent {
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

  kInferenceWorkloadDone,

  kNumEvent,
};

std::ostream& operator<<(std::ostream &os, ctrl::CtrlEvent event);
std::ostream& operator<<(std::ostream &os, ctrl::CtrlMsgEntry msg);


template <typename T>
class BasicMessageQueue {
  static_assert(std::is_trivially_copyable<T>::value, 
                "T must be trivially copyable");
 public:
  BasicMessageQueue(bool is_server, const std::string &name, 
                    bip::managed_shared_memory &bip_shm) 
                    : bip_shm_(bip_shm) {
    this->name_ = "basic_mq_" + name;
    if (is_server) {
      mq_ = bip_shm.find_or_construct<bip_deque<T>>(name_.c_str())
          (bip_shm.get_segment_manager());
      mut_ = bip_shm.find_or_construct<bip_mutex>(
          (name_ + "_mutex").c_str())();
      get_cond_ = bip_shm.find_or_construct<bip_cond>(
          (name_ + "_get_cond").c_str())();
      put_cond_ = bip_shm.find_or_construct<bip_cond>(
          (name_ + "_put_cond").c_str())();
    } else {
      mq_ = bip_shm.find<bip_deque<T>>(name_.c_str()).first;
      mut_ = bip_shm.find<bip_mutex>((name_ + "_mutex").c_str()).first;
      get_cond_ = bip_shm.find<bip_cond>((name_ + "_get_cond").c_str()).first;
      put_cond_ = bip_shm.find<bip_cond>((name_ + "_put_cond").c_str()).first;
    }
  }

  void Put(const T &msg) {
    bip::scoped_lock<bip_mutex> lock{*mut_};
    mq_->push_back(msg);
    get_cond_->notify_one();
  }

  void Put(const char* data) {
    Put(*reinterpret_cast<T*>(data));
  }

  void BlockPut(const T &msg) {
    bip::scoped_lock<bip_mutex> lock{*mut_};
    mq_->push_back(msg);
    get_cond_->notify_one();

    put_cond_->wait(lock, [this] { return mq_->empty(); });
    return;
  }

  void BlockPut(const void* data) {
    BlockPut(*reinterpret_cast<T*>(data));
  }

  T BlockGet() {
    bip::scoped_lock<bip_mutex> lock{*mut_};
    get_cond_->wait(lock, [this] { return !mq_->empty(); });
    T msg = mq_->front();
    mq_->pop_front();
    if (mq_->empty()) {
      put_cond_->notify_one();
    }
    return msg;
  }
  
  bool TryGet(T &msg) {
    bip::scoped_lock<bip_mutex> lock{*mut_};
    if (mq_->empty()) {
      return false;
    }
    msg = mq_->front();
    mq_->pop_front();

    if (mq_->empty()) {
      put_cond_->notify_one();
    }
    return true;
  }

  bool TimedGet(uint32_t timeout_ms, T &msg) {
    bip::scoped_lock<bip_mutex> lock{*mut_};
    get_cond_->wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                    [=] { return !mq_->empty(); });
    if (mq_->empty()) {
      return false;
    }
    msg = mq_->front();
    mq_->pop_front();

    if (mq_->empty()) {
      put_cond_->notify_one();
    }
    return true;
  }

 private:
  std::string name_;
  bip_deque<T> *mq_{nullptr};
  bip_mutex *mut_{nullptr};
  bip_cond *get_cond_{nullptr}, *put_cond_{nullptr};
  bip::managed_shared_memory &bip_shm_;
};

class InfTraMessageQueue {
 public: 
  enum class Direction {
    kInf2Tra,
    kTra2Inf,
    kNumDirection,
  };

  InfTraMessageQueue(bool is_server, int train_world_size,
                     bip::managed_shared_memory &bip_shm,
                     bip::scoped_lock<bip_mutex> &lock);

  CtrlMsgEntry BlockGet(Direction direction, int id);
  bool TryGet(Direction direction, int id, CtrlMsgEntry &msg);
  bool TimedGet(uint32_t timeout_ms, Direction direction, int id,
                CtrlMsgEntry &msg);

  std::pair<int, CtrlMsgEntry> BlockGetFromAny(Direction direction);
  bool TryGetFromAny(Direction direction, int &id, CtrlMsgEntry &msg);
  bool TimedGetFromAny(uint32_t timeout_ms, Direction direction, 
                       int &id, CtrlMsgEntry &msg);

  void Put(const CtrlMsgEntry &msg, Direction direction, int id);
  void PutAll(const CtrlMsgEntry &msg, Direction direction);
  void PutAll(const std::vector<CtrlMsgEntry> &msg, Direction direction);

  void Clear();

 private:
  std::string GetMqName(Direction direction, int id);
  std::string GetMqMutexName(Direction direction, int id);
  std::string GetMqCondName(Direction direction, int id);
  std::string GetMqGroupMutexName(Direction direction);
  std::string GetMqGroupCondName(Direction direction);

  template<typename T>
  void PutAllImpl(const T &msg, Direction direction);

  int message_queue_num_;
  std::array<bip_deque<CtrlMsgEntry>*, MAX_DEVICE_NUM> 
      inf_tra_mqs_[static_cast<int>(Direction::kNumDirection)] = {nullptr};

  std::array<bip_mutex*, MAX_DEVICE_NUM> 
      inf_tra_mq_muts_[static_cast<int>(Direction::kNumDirection)] = {nullptr};

  std::array<bip_cond*, MAX_DEVICE_NUM>
      inf_tra_mq_conds_[static_cast<int>(Direction::kNumDirection)] = {nullptr};
  
  bip_mutex 
      *inf_tra_mq_group_muts_[static_cast<int>(Direction::kNumDirection)] = {nullptr};
  bip_cond* 
      inf_tra_mq_group_conds_[static_cast<int>(Direction::kNumDirection)] = {nullptr};

  bip::managed_shared_memory &bip_shm_;
};

std::ostream &operator<<(std::ostream &os, InfTraMessageQueue::Direction direction);

}
}