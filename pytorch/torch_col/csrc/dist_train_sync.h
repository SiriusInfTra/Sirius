#pragma once

#include <common/inf_tra_comm/bip_helper.h>
#include <pthread.h>

namespace torch_col {

class DistTrainSync {
 public:
  static void Init();
  static void WaitBarrier();

  DistTrainSync();
  ~DistTrainSync();

 private:
  static std::unique_ptr<DistTrainSync> dist_train_sync_;

  pthread_barrier_t *barrier_;
  colserve::bip_named_sem *sem_; // semaphore for init
  std::string shm_name_;
  std::string sem_name_;
  colserve::bip::managed_shared_memory bip_shm_;
};


}