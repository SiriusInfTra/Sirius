#include <torch_col/csrc/config.h>
#include <torch_col/csrc/dist_train_sync.h>

#include <common/log_as_glog_sta.h>
#include <common/util.h>
#include <common/inf_tra_comm/communicator.h>

#include <thread>


namespace torch_col {

std::unique_ptr<DistTrainSync> DistTrainSync::dist_train_sync_ = nullptr;
bool DistTrainSync::initialized_ = false;

void DistTrainSync::Init() {
  CHECK(DistTrainSync::dist_train_sync_ == nullptr)
      << "DistTrainSync has been initialized";
  DistTrainSync::dist_train_sync_ = std::make_unique<DistTrainSync>();
  initialized_ = true;
  LOG(INFO) << "DistTrainSync initialized";
}

bool DistTrainSync::IsInitialized() {
  return initialized_;
}

void DistTrainSync::WaitBarrier() {
  CHECK(DistTrainSync::dist_train_sync_ != nullptr);

  auto err = pthread_barrier_wait(dist_train_sync_->barrier_);
  CHECK(err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD) 
      << "pthread_barrier_wait err " << err;
}

void DistTrainSync::Send(int dst, const std::string &msg) {
  CHECK(DistTrainSync::dist_train_sync_ != nullptr);

  int src = TorchColConfig::GetTrainRank();
  for (int i = 0; i < msg.size(); ) {
    int j = i + kMsgBufSize >= msg.size() ? msg.size() : i + kMsgBufSize;

    std::array<char, kMsgBufSize> buf{0};
    std::copy(msg.data() + i, msg.data() + j, buf.begin());

    dist_train_sync_->intra_train_str_mqs_[src][dst]->BlockPut(buf);
    i = j;
  }
}

std::string DistTrainSync::Recv(int src) {
  std::vector<char> ret;
  int rank = TorchColConfig::GetTrainRank();
  while (true) {
    auto buf = dist_train_sync_->intra_train_str_mqs_[src][rank]->BlockGet();
    auto it = std::find(buf.begin(), buf.end(), 0);
    if (it != buf.end()) {
      ret.insert(ret.end(), buf.begin(), it);
      break;
    } else {
      ret.insert(ret.end(), buf.begin(), buf.end());
    }
  }
  return std::string(ret.begin(), ret.end());
}

DistTrainSync::DistTrainSync() {

  namespace bip = colserve::bip;
  CHECK(TorchColConfig::IsConfigured());
  shm_name_ = colserve::GetDefaultShmNamePrefix() + "_dist_train_sync";

  // semaphore for init
  auto sem_name = colserve::GetDefaultShmNamePrefix() + "_dist_train_sync_sem";
  colserve::bip_named_sem init_sem = bip::named_semaphore{
      bip::open_or_create, sem_name.c_str(), 0};

  int train_world_size = TorchColConfig::GetTrainWorldSize();
  if (TorchColConfig::IsTrainMaster()) {
    bip::shared_memory_object::remove(shm_name_.c_str());
    bip_shm_ = bip::managed_shared_memory{bip::create_only,
                                          shm_name_.c_str(), 1024 * 1024};
    auto atomic_init = [&]() {
      DLOG(INFO) << "dist train sync init rank " 
                 << TorchColConfig::GetTrainRank();
      barrier_ = 
          bip_shm_.find_or_construct<pthread_barrier_t>("barrier")();
      for (int i = 0; i < train_world_size; i++) {
        for (int j = 0; j < train_world_size; j++) {
          intra_train_str_mqs_[i][j].reset(new dist_train_str_mq_t(
              true, GetMqName("str-msg", i, j), bip_shm_));
        }
      }

      pthread_barrierattr_t attr;
      CHECK_EQ(pthread_barrierattr_init(&attr), 0);
      CHECK_EQ(pthread_barrierattr_setpshared(&attr, PTHREAD_PROCESS_SHARED), 0);
      CHECK_EQ(pthread_barrier_init(barrier_, &attr, 
                                    TorchColConfig::GetTrainWorldSize()), 0);
    };
    bip_shm_.atomic_func(atomic_init);

    for (int i = 1; i < train_world_size; i++) {
      init_sem.post();
    }
  } else {
    bool succ = init_sem.timed_wait(
        std::chrono::system_clock::now() 
        + std::chrono::seconds(10)
    );
    if (!succ) {
      LOG(FATAL) << "dist train sync init timeout, "
                 << str(boost::format("`rm -f /dev/shm/sem.%s` and retry") % sem_name);
    }
    bip_shm_ = bip::managed_shared_memory{bip::open_only, shm_name_.c_str()};
    auto atomic_init = [&]() {
      DLOG(INFO) << "dist train sync init rank " << TorchColConfig::GetTrainRank();

      barrier_ = bip_shm_.find<pthread_barrier_t>("barrier").first;
      for (int i = 0; i < train_world_size; i++) {
        for (int j = 0; j < train_world_size; j++) {
          intra_train_str_mqs_[i][j].reset(new dist_train_str_mq_t(
              false, GetMqName("str-msg", i, j), bip_shm_));
        }
      }
    };
    bip_shm_.atomic_func(atomic_init);
  }

  // for avoiding name conflict
  intra_train_custom_mq_dict_.insert({"str-msg", nullptr});

  auto err = pthread_barrier_wait(barrier_);

  if (TorchColConfig::IsTrainMaster()) {
    bip::shared_memory_object::remove(sem_name.c_str());
  }

  CHECK(err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD)
      << "pthread_barrier_wait err " << err;
  DLOG(INFO) << "[DistTrainSync | Rank " << TorchColConfig::GetTrainRank() 
            <<  "] initialized";
}

DistTrainSync::~DistTrainSync() {

}

} // namespace torch_col