#include <common/inf_tra_comm/message_queue.h>
#include <common/log_as_glog_sta.h>

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/format.hpp>
#include <chrono>


namespace colserve {
namespace ctrl {

InfTraMessageQueue::InfTraMessageQueue(bool is_server,
                                       int train_world_size, 
                                       bip::managed_shared_memory &bip_shm,
                                       bip::scoped_lock<bip_mutex> &lock) 
    : bip_shm_(bip_shm) {
  if (is_server) {
    for (int i = 0; i < static_cast<int>(Direction::kNumDirection); i++) {
      for (int j = 0; j < train_world_size; j++) {
        inf_tra_mqs_[i][j] = bip_shm_.find_or_construct<bip_deque<CtrlMsgEntry>>
            (GetMqName(static_cast<Direction>(i), j).c_str())
            (bip_shm.get_segment_manager());
        
        inf_tra_mq_muts_[i][j] = bip_shm_.find_or_construct<bip_mutex>
            (GetMqMutexName(static_cast<Direction>(i), j).c_str())();

        inf_tra_mq_conds_[i][j] = bip_shm_.find_or_construct<bip_cond>
            (GetMqCondName(static_cast<Direction>(i), j).c_str())();
      }
    }
  } else {
    for (int i = 0; i < static_cast<int>(Direction::kNumDirection); i++) {
      for (int j = 0; j < train_world_size; j++) {
        inf_tra_mqs_[i][j] = bip_shm_.find<bip_deque<CtrlMsgEntry>>
            (GetMqName(static_cast<Direction>(i), j).c_str()).first;
        
        inf_tra_mq_muts_[i][j] = bip_shm_.find<bip_mutex>
            (GetMqMutexName(static_cast<Direction>(i), j).c_str()).first;

        inf_tra_mq_conds_[i][j] = bip_shm_.find<bip_cond>
            (GetMqCondName(static_cast<Direction>(i), j).c_str()).first;
      }
    }
  }
}

CtrlMsgEntry InfTraMessageQueue::BlockGet(Direction direction, int id) {
  bip::scoped_lock<bip_mutex> 
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};

  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];

  inf_tra_mq_conds_[static_cast<int>(direction)][id]->wait(lock, 
      [=]() { return !inf_tra_mq->empty(); });

  auto ret = inf_tra_mq->front();
  inf_tra_mq->pop_front();
  return ret;
}

bool InfTraMessageQueue::TryGet(CtrlMsgEntry &msg, Direction direction, int id) {
  bip::scoped_lock<bip_mutex>
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};
  
  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  if (inf_tra_mq->empty()) {
    return false;
  }

  msg = inf_tra_mq->front();
  inf_tra_mq->pop_front();
  return true;
}

bool InfTraMessageQueue::TimedGet(CtrlMsgEntry &msg, uint32_t timeout_ms, 
                                  Direction direction, int id) {
  bip::scoped_lock<bip_mutex>
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};
  
  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  inf_tra_mq_conds_[static_cast<int>(direction)][id]->wait_for(lock, 
      std::chrono::milliseconds(timeout_ms),
      [=]() { return !inf_tra_mq->empty(); });

  if (inf_tra_mq->empty()) {
    return false;
  }

  msg = inf_tra_mq->front();
  inf_tra_mq->pop_front();
  return true;
}

void InfTraMessageQueue::Put(CtrlMsgEntry msg, Direction direction, int id) {
  bip::scoped_lock<bip_mutex>
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};

  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  inf_tra_mq->push_back(msg);
  inf_tra_mq_conds_[static_cast<int>(direction)][id]->notify_one();
}

std::string InfTraMessageQueue::GetMqName(Direction direction, int id) {
  switch (direction) {
  case Direction::kInf2Tra:
    return str(boost::format("inf2tra-mq-%d") % id);
  case Direction::kTra2Inf:
    return str(boost::format("tra2inf-mq-%d") % id);
  default:
    LOG(FATAL) << "unknown Inf-Tra MQ direction " << static_cast<int>(direction);
    return "";
  }
}

std::string InfTraMessageQueue::GetMqMutexName(Direction direction, int id) {
  switch (direction) {
  case Direction::kInf2Tra:
    return str(boost::format("inf2tra-mq-mut-%d") %id);
  case Direction::kTra2Inf:
    return str(boost::format("tra2inf-mq-mut-%d") % id);
  default:
    LOG(FATAL) << "unknown Inf-Tra MQ direction " << static_cast<int>(direction);
    return "";
  }
}

std::string InfTraMessageQueue::GetMqCondName(Direction direction, int id) {
  switch (direction) {
  case Direction::kInf2Tra:
    return str(boost::format("inf2tra-mq-cond-%d") % id);
  case Direction::kTra2Inf:
    return str(boost::format("tra2inf-mq-cond-%d") % id);
  default:
    LOG(FATAL) << "unknown Inf-Tra MQ direction " << static_cast<int>(direction);
    return ""; 
  }
}

}
}