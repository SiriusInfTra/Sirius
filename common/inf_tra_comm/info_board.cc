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
    infer_info_mut_ = bip_shm_.find_or_construct<bip_mutex>
        (GetInferInfoMutexName().c_str())();

    for (int i = 0; i < train_world_size; i++) {
      train_infos_[i] = bip_shm_.find_or_construct<TrainInfo>
        (GetTrainInfoName(i).c_str())();
      train_info_muts_[i] = bip_shm_.find_or_construct<bip_mutex>
          (GetTrainInfoMutexName(i).c_str())();

      InvalidateTrainInfo(i);
    }
  } else {
    infer_info_ = bip_shm_.find<InferInfo>
        (GetInferInfoName().c_str()).first;
    infer_info_mut_ = bip_shm_.find<bip_mutex>
        (GetInferInfoMutexName().c_str()).first;

    for (int i = 0; i < train_world_size; i++) {
      train_infos_[i] = bip_shm_.find<TrainInfo>
          (GetTrainInfoName(i).c_str()).first;
      train_info_muts_[i] = bip_shm_.find<bip_mutex>
          (GetTrainInfoMutexName(i).c_str()).first;

      CHECK(!IsTrainInfoValid(i)) << train_infos_[i]->train_pid;
    }
  }
}

void InfTraInfoBoard::SetInferInfo(std::function<void(InferInfo*)> fn) {
  bip::scoped_lock lock{*infer_info_mut_};
  fn(infer_info_);
}

void InfTraInfoBoard::SetTrainInfo(int id, std::function<void(TrainInfo*)> fn) {
  bip::scoped_lock lock{*train_info_muts_[id]};
  fn(train_infos_[id]);
}

void InfTraInfoBoard::SetTrainInfo(
    int id, std::optional<pid_t> pid, 
    std::optional<int> rank, std::optional<int> world_size,
    std::optional<int> init_batch_size,
    std::optional<int> current_batch_size) {
  bip::scoped_lock lock{*train_info_muts_[id]};
  if (pid.has_value()) {
    train_infos_[id]->train_pid = pid.value();
  }
  if (rank.has_value()) {
    train_infos_[id]->train_rank = rank.value();
  }
  if (world_size.has_value()) {
    train_infos_[id]->train_world_size = world_size.value();
  }
  if (init_batch_size.has_value()) {
    train_infos_[id]->init_batch_size = init_batch_size.value();
  }
  if (current_batch_size.has_value()) {
    train_infos_[id]->current_batch_size = current_batch_size.value();
  }
}

} // namespace ctrl
} // namespace colserve