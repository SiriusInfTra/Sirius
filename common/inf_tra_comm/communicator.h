#pragma once

#include <common/inf_tra_comm/bip_helper.h>
#include <common/inf_tra_comm/message_queue.h>
#include <common/inf_tra_comm/info_board.h>


namespace colserve {
namespace ctrl {

class InfTraCommunicator {
 public:
  static void Init(bool is_server, bool cleanup, int training_world_size);
  static InfTraMessageQueue* GetMQ();
  static InfTraInfoBoard* GetIB();

  InfTraCommunicator(const std::string &shm_name, bool is_server, 
                     bool cleanup, int training_world_size);
  ~InfTraCommunicator();

 private:
  static std::unique_ptr<InfTraCommunicator> communicator_;

  std::string shm_name_;
  bool is_server_;
  bip_mutex *mut_;

  std::unique_ptr<InfTraMessageQueue> message_queue_;
  std::unique_ptr<InfTraInfoBoard> info_board_;

  bip::managed_shared_memory bip_shm_;
};


}
}