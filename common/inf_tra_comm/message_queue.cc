#include <common/inf_tra_comm/message_queue.h>
#include <common/log_as_glog_sta.h>

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/format.hpp>
#include <chrono>


namespace colserve {
namespace ctrl {

std::ostream& operator<<(std::ostream&os, ctrl::CtrlEvent event) {
  switch (event) {
  // status event
  case ctrl::CtrlEvent::kTrainStart:
    os << "CtrlEvent::kTrainStart"; 
    return os;
  case ctrl::CtrlEvent::kTrainEnd:
    os << "CtrlEvent::kTrainEnd"; 
    return os;
  case ctrl::CtrlEvent::kInterruptTrainDone:
    os << "CtrlEvent::kInterruptTrainDone";
    return os;
  case ctrl::CtrlEvent::kResumeTrainDone:
    os << "CtrlEvent::kResumeTrainDone";
    return os;

  // cmd event
  case ctrl::CtrlEvent::kInterruptTrain:
    os << "CtrlEvent::kInterruptTrain";
    return os;
  case ctrl::CtrlEvent::kResumeTrain:
    os << "CtrlEvent::kResumeTrain";
    return os;
  default:
    LOG(FATAL) << "unknown ctrl::CtrlEvent" << static_cast<int>(event);
    return os;
  }
}

std::ostream& operator<<(std::ostream& os, ctrl::CtrlMsgEntry msg) {
  os << "CtrlMsgEntry{id=" << msg.id << ", event=" << msg.event 
     << ", value=" << msg.value << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, 
                         InfTraMessageQueue::Direction direction) {
  switch (direction) {
  case InfTraMessageQueue::Direction::kInf2Tra:
    os << "Direction::kInf2Tra";
    return os;
  case InfTraMessageQueue::Direction::kTra2Inf:
    os << "Direction::kTra2Inf";
    return os;
  default:
    LOG(FATAL) << "unknown InfTraMessageQueue::Direction " 
               << static_cast<int>(direction);
    return os;
  }
}

InfTraMessageQueue::InfTraMessageQueue(bool is_server,
                                       int train_world_size, 
                                       bip::managed_shared_memory &bip_shm,
                                       bip::scoped_lock<bip_mutex> &lock) 
    : message_queue_num_(train_world_size), bip_shm_(bip_shm) {
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
      inf_tra_mq_group_muts_[i] = bip_shm_.find_or_construct<bip_mutex>
          (GetMqGroupMutexName(static_cast<Direction>(i)).c_str())();
      inf_tra_mq_group_conds_[i] = bip_shm_.find_or_construct<bip_cond>
          (GetMqGroupCondName(static_cast<Direction>(i)).c_str())();
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
      inf_tra_mq_group_muts_[i] = bip_shm_.find<bip_mutex>
          (GetMqGroupMutexName(static_cast<Direction>(i)).c_str()).first;
      inf_tra_mq_group_conds_[i] = bip_shm_.find<bip_cond>
          (GetMqGroupCondName(static_cast<Direction>(i)).c_str()).first;
    }
  }
}

CtrlMsgEntry InfTraMessageQueue::BlockGet(Direction direction, int id) {
  bip::scoped_lock<bip_mutex> 
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};

  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  CHECK(inf_tra_mq != nullptr); 

  inf_tra_mq_conds_[static_cast<int>(direction)][id]->wait(lock, 
      [=]() { return !inf_tra_mq->empty(); });

  auto ret = inf_tra_mq->front();
  inf_tra_mq->pop_front();
  DLOG(INFO) << "[InfTra MQ] BlockGet " << ret << " from " 
            << GetMqName(direction, id);
  return ret;
}

bool InfTraMessageQueue::TryGet(Direction direction, int id,
                                CtrlMsgEntry &msg) {
  bip::scoped_lock<bip_mutex>
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};
  
  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  CHECK(inf_tra_mq != nullptr);

  if (inf_tra_mq->empty()) {
    return false;
  }

  msg = inf_tra_mq->front();
  inf_tra_mq->pop_front();
  DLOG(INFO) << "[InfTra MQ] TryGet " << msg << " from " 
            << GetMqName(direction, id); 
  return true;
}

bool InfTraMessageQueue::TimedGet(uint32_t timeout_ms, 
                                  Direction direction, int id,
                                  CtrlMsgEntry &msg) {
  bip::scoped_lock<bip_mutex>
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};
  
  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  CHECK(inf_tra_mq != nullptr);

  inf_tra_mq_conds_[static_cast<int>(direction)][id]->wait_for(lock, 
      std::chrono::milliseconds(timeout_ms),
      [=]() { return !inf_tra_mq->empty(); });

  if (inf_tra_mq->empty()) {
    return false;
  }

  msg = inf_tra_mq->front();
  inf_tra_mq->pop_front();
  DLOG(INFO) << "[InfTra MQ] TimedGet " << msg << " from " 
            << GetMqName(direction, id);
  return true;
}

std::pair<int, CtrlMsgEntry> 
InfTraMessageQueue::BlockGetFromAny(Direction direction) {
  bip::scoped_lock 
      group_lock{*inf_tra_mq_group_muts_[static_cast<int>(direction)]};

  inf_tra_mq_group_conds_[static_cast<int>(direction)]->wait(group_lock, 
      [=]() {
        for (int i = 0; i < this->message_queue_num_; i++) {
          auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][i];
          CHECK(inf_tra_mq != nullptr);
          if (!inf_tra_mq->empty()) { return true; }
        }
        return false;
      });

  for (int i = 0; i < this->message_queue_num_; i++) {
    auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][i];
    CHECK(inf_tra_mq != nullptr);

    bip::scoped_lock 
        lock{*inf_tra_mq_muts_[static_cast<int>(direction)][i]};
    if (!inf_tra_mq->empty()) {
      auto ret = std::make_pair(i, inf_tra_mq->front());
      inf_tra_mq->pop_front();
      DLOG(INFO) << "[InfTra MQ] BlockGetFromAny " << ret.second << " from " 
                << GetMqName(direction, i);
      return ret;
    }
  }

  LOG(FATAL) << "should not reach here";
  return {};
}

void InfTraMessageQueue::Put(const CtrlMsgEntry &msg, Direction direction, int id) {
  bip::scoped_lock<bip_mutex>
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][id]};

  auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][id];
  CHECK(inf_tra_mq != nullptr);

  inf_tra_mq->push_back(msg);
  inf_tra_mq_conds_[static_cast<int>(direction)][id]->notify_one();
  inf_tra_mq_group_conds_[static_cast<int>(direction)]->notify_one();

  DLOG(INFO) << "[InfTra MQ] Put " << msg << " to " 
            << GetMqName(direction, id);
}

void InfTraMessageQueue::PutAll(const CtrlMsgEntry &msg, Direction direction) {
  for (int i = 0; i < message_queue_num_; i++) {
    bip::scoped_lock
      lock{*inf_tra_mq_muts_[static_cast<int>(direction)][i]};

    auto inf_tra_mq = inf_tra_mqs_[static_cast<int>(direction)][i];
    CHECK(inf_tra_mq != nullptr);
    inf_tra_mq->push_back(msg);
    inf_tra_mq_conds_[static_cast<int>(direction)][i]->notify_one();
  }
  inf_tra_mq_group_conds_[static_cast<int>(direction)]->notify_one();

  DLOG(INFO) << "[InfTra MQ] PutAll " << msg << " to all " 
            << direction;
}

void InfTraMessageQueue::Clear() {
  for (int i = 0; i < static_cast<int>(Direction::kNumDirection); i++) {
    for (int j = 0; j < message_queue_num_; j++) {
      bip::scoped_lock<bip_mutex>
          lock{*inf_tra_mq_muts_[i][j]};
      auto inf_tra_mq = inf_tra_mqs_[i][j];
      CHECK(inf_tra_mq != nullptr);
      inf_tra_mq->clear();
    }
  }
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

std::string InfTraMessageQueue::GetMqGroupMutexName(Direction direction) {
  switch (direction) {
  case Direction::kInf2Tra:
    return "inf2tra-mq-group-mut";
  case Direction::kTra2Inf:
    return "tra2inf-mq-group-mut";
  default:
    LOG(FATAL) << "unknown Inf-Tra MQ direction " << static_cast<int>(direction);
    return "";
  }
}

std::string InfTraMessageQueue::GetMqGroupCondName(Direction direction) {
  switch (direction) {
  case Direction::kInf2Tra:
    return "inf2tra-mq-group-cond";
  case Direction::kTra2Inf:
    return "tra2inf-mq-group-cond";
  default:
    LOG(FATAL) << "unknown Inf-Tra MQ direction " << static_cast<int>(direction);
    return "";
  }
}

}
}