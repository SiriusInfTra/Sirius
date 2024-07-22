#include <common/inf_tra_comm/info_board.h>


namespace colserve {
namespace ctrl {

InfTraInfoBoard::InfTraInfoBoard(bool is_server, int train_world_size,
                                 bip::managed_shared_memory &bip_shm,
                                 bip::scoped_lock<bip_mutex> &lock)
    : bip_shm_(bip_shm) {
  if (is_server) {
    infer_info_ = bip_shm_.find_or_construct<InferInfo>
        (GetInferInfoName().c_str())();
    
    for (int i = 0; i < train_world_size; i++) {
      train_infos_[i] = bip_shm_.find_or_construct<TrainInfo>
        (GetTrainInfoName(i).c_str())();
    }
  } else {
    infer_info_ = bip_shm_.find<InferInfo>
        (GetInferInfoName().c_str()).first;

    for (int i = 0; i < train_world_size; i++) {
      train_infos_[i] = bip_shm_.find<TrainInfo>
          (GetTrainInfoName(i).c_str()).first;
    }
  }

}

}
}