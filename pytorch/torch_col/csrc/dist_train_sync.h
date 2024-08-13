#pragma once

#include <common/inf_tra_comm/bip_helper.h>
#include <common/inf_tra_comm/message_queue.h>
#include <pthread.h>

namespace torch_col {

class DistTrainSync {
 public:
  static void Init();
  static void WaitBarrier();
  static void Send(int dst, const std::string &msg);
  static std::string Recv(int src);

  DistTrainSync();
  ~DistTrainSync();

 private:
  static std::unique_ptr<DistTrainSync> dist_train_sync_;

  constexpr static size_t kMsgBufSize = 256;
  using dist_train_mq_t = 
      colserve::ctrl::BasicMessageQueue<std::array<char, kMsgBufSize>>;

  std::string GetMqName(int src, int dst) {
    return str(boost::format("dist-train-mq-%d-%d") % src % dst);
  }

  pthread_barrier_t *barrier_;
  std::unique_ptr<dist_train_mq_t>
      intra_train_mqs_[colserve::MAX_DEVICE_NUM][colserve::MAX_DEVICE_NUM]{nullptr};

  colserve::bip_named_sem *sem_; // semaphore for init
  std::string shm_name_;
  std::string sem_name_;
  colserve::bip::managed_shared_memory bip_shm_;
};


}