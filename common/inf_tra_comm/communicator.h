#pragma once

#include <common/inf_tra_comm/bip_helper.h>
#include <common/inf_tra_comm/message_queue.h>
#include <common/inf_tra_comm/shared_info.h>


namespace colserve {
namespace ctrl {

enum {
  kTraRank_0 = 0,
  kTraRank_1 = 1,
  kTraRank_2 = 2,
  kTraRank_3 = 3,
};

class InfTraCommunicator {
 public:
  static void Init(bool is_server, bool cleanup, int train_world_size);
  static bool IsInitialized() { return communicator_ != nullptr; }
  static InfTraMessageQueue* GetMQ();
  static InfTraSharedInfo* GetIB();

  static int GetTrainWorldSize() {
    return GetIB()->GetTrainInfoUnsafe(kTraRank_0)->train_world_size; 
  }
  static pid_t GetTrainPID(int rank) {
    return GetIB()->GetTrainInfoUnsafe(rank)->train_pid;
  }
  static std::vector<pid_t> GetTrainPIDs() {
    return GetIB()->GetTrainPIDs();
  }

  InfTraCommunicator(const std::string &shm_name, bool is_server, 
                     bool cleanup, int train_world_size);
  ~InfTraCommunicator();

 private:
  static std::unique_ptr<InfTraCommunicator> communicator_;

  std::string shm_name_;
  bool is_server_;
  bip_mutex *mut_;

  std::unique_ptr<InfTraMessageQueue> message_queue_;
  std::unique_ptr<InfTraSharedInfo> shared_info_;

  bip::managed_shared_memory bip_shm_;
};


}
}