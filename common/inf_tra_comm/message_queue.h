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

  void Clear();

 private:
  std::string GetMqName(Direction direction, int id);
  std::string GetMqMutexName(Direction direction, int id);
  std::string GetMqCondName(Direction direction, int id);
  std::string GetMqGroupMutexName(Direction direction);
  std::string GetMqGroupCondName(Direction direction);


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