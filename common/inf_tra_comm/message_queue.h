#pragma once

#include <common/controlling.h>
#include <common/inf_tra_comm/bip_helper.h>
#include <common/util.h>

#include <boost/format.hpp>
#include <array>

namespace colserve {
namespace ctrl {

class InfTraMessageQueue {
 public: 
  enum class Direction {
    kInf2Tra,
    kTra2Inf,
    kNumDirection,
  };

  InfTraMessageQueue(bool is_server, int training_world_size,
                     bip::managed_shared_memory &bip_shm,
                     bip::scoped_lock<bip_mutex> &lock);

  CtrlMsgEntry BlockGet(Direction direction, int id);
  bool TryGet(CtrlMsgEntry &msg, Direction direction, int id);
  bool TimedGet(CtrlMsgEntry &msg, uint32_t timeout_ms, Direction direction, int id);

  void Put(CtrlMsgEntry msg, Direction direction, int id);

 private:
  std::string GetMqName(Direction direction, int id);
  std::string GetMqMutexName(Direction direction, int id);
  std::string GetMqCondName(Direction direction, int id);

  std::array<bip_deque<CtrlMsgEntry>*, MAX_DEVICE_NUM> 
      inf_tra_mqs_[static_cast<int>(Direction::kNumDirection)] = {nullptr};

  std::array<bip_mutex*, MAX_DEVICE_NUM> 
      inf_tra_mq_muts_[static_cast<int>(Direction::kNumDirection)] = {nullptr};

  std::array<bip_cond*, MAX_DEVICE_NUM>
      inf_tra_mq_conds_[static_cast<int>(Direction::kNumDirection)] = {nullptr};

  bip::managed_shared_memory &bip_shm_;
};

}
}