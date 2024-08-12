#include <common/log_as_glog_sta.h>
#include <common/util.h>

#include <torch_col/csrc/config.h>
#include <torch_col/csrc/dist_train_sync.h>


namespace torch_col {

std::unique_ptr<DistTrainSync> DistTrainSync::dist_train_sync_ = nullptr; 

void DistTrainSync::Init() {
  CHECK(DistTrainSync::dist_train_sync_ == nullptr)
      << "DistTrainSync has been initialized";
  DistTrainSync::dist_train_sync_ = std::make_unique<DistTrainSync>();
  LOG(INFO) << "DistTrainSync initialized";
}

void DistTrainSync::WaitBarrier() {
  CHECK(DistTrainSync::dist_train_sync_ != nullptr);
  CHECK_EQ(pthread_barrier_wait(dist_train_sync_->barrier_), 0);
}

DistTrainSync::DistTrainSync() {
  namespace bip = colserve::bip;
  CHECK(TorchColConfig::IsConfigured());
  shm_name_ = colserve::GetDefaultShmNamePrefix() + "_dist_train_sync";
  sem_name_ = colserve::GetDefaultShmNamePrefix() + "_dist_train_sync_sem";

  sem_ = new bip::named_semaphore{bip::open_or_create, 
                                  sem_name_.c_str(), 0};

  if (TorchColConfig::GetTrainRank() == 0) {
    bip::shared_memory_object::remove(shm_name_.c_str());
    bip_shm_ = bip::managed_shared_memory{bip::create_only,
                                          shm_name_.c_str(), 65536};
    auto atomic_init = [&]() {
      barrier_ = 
          bip_shm_.find_or_construct<pthread_barrier_t>("barrier")();
      pthread_barrierattr_t attr;
      CHECK_EQ(pthread_barrierattr_init(&attr), 0);
      CHECK_EQ(pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED), 0);
      CHECK_EQ(pthread_barrier_init(barrier_, &attr, 
                                    TorchColConfig::GetTrainWorldSize()), 0);
    };
    atomic_init();
    sem_->post();
  } else {
    sem_->wait();
    auto atomic_init = [&]() {
      barrier_ = bip_shm_.find<pthread_barrier_t>("barrier").first;
    };
  }
  WaitBarrier();
}

DistTrainSync::~DistTrainSync() {
  delete sem_;
}

} // namespace torch_col