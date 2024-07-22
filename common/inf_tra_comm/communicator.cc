#include <common/inf_tra_comm/communicator.h>


namespace colserve {
namespace ctrl {

std::unique_ptr<InfTraCommunicator> InfTraCommunicator::communicator_ = nullptr;

void InfTraCommunicator::Init(bool is_server, bool cleanup, 
                              int training_world_size) {
  if (communicator_ != nullptr) {
    LOG(FATAL) << "InfTraCommunicator has already been initialized.";
  }
  communicator_ = std::make_unique<InfTraCommunicator>(
      GetDefaultShmNamePrefix() + "_inf_tra_comm", 
      is_server, cleanup, training_world_size);
}

InfTraMessageQueue* InfTraCommunicator::GetMQ() {
  if (communicator_ == nullptr) {
    LOG(FATAL) << "InfTraCommunicator has not been initialized.";
  }
  return communicator_->message_queue_.get();
}

InfTraCommunicator::InfTraCommunicator(const std::string &shm_name, 
                                       bool is_server, bool cleanup,
                                       int training_world_size)
    : shm_name_(shm_name), is_server_(is_server) {
  if (is_server && cleanup) {
    bip::shared_memory_object::remove(shm_name_.c_str());
  }

  if (is_server) {
    bip_shm_ = bip::managed_shared_memory{bip::open_or_create, 
                                          shm_name_.c_str(), 65536};
    mut_ = bip_shm_.find_or_construct<bip_mutex>("shm_mutex")();
  } else {
    bip_shm_ = bip::managed_shared_memory{bip::open_only, shm_name_.c_str()};
    mut_ = bip_shm_.find<bip_mutex>("shm_mutex").first;
  }

  // Initialize message queue and information board
  bip::scoped_lock<bip_mutex> lock{*mut_};
  message_queue_ = std::make_unique<InfTraMessageQueue>(
      is_server, training_world_size, bip_shm_, lock);
  info_board_ = std::make_unique<InfTraInfoBoard>(
      is_server, training_world_size, bip_shm_, lock);
}

InfTraCommunicator::~InfTraCommunicator() {
  if (is_server_) {
    bip::shared_memory_object::remove(shm_name_.c_str());
  }
}

}
}