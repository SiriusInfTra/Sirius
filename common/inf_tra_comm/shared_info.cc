#include <common/inf_tra_comm/shared_info.h>

#include <boost/range/irange.hpp>


namespace colserve {
namespace ctrl {

InfTraSharedInfo::InfTraSharedInfo(bool is_server, int train_world_size,
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

void InfTraSharedInfo::SetInferInfo(std::function<void(InferInfo*)> fn) {
  bip::scoped_lock lock{*infer_info_mut_};
  fn(infer_info_);
}

void InfTraSharedInfo::SetTrainInfo(int id, std::function<void(TrainInfo*)> fn) {
  bip::scoped_lock lock{*train_info_muts_[id]};
  fn(train_infos_[id]);
}

void InfTraSharedInfo::SetTrainInfo(
    int id, 
    std::optional<pid_t> pid, 
    std::optional<int> rank, 
    std::optional<int> world_size,
    std::optional<int> init_batch_size,
    std::optional<int> current_batch_size,
    std::optional<const char*> model_name) {
  bip::scoped_lock lock{*train_info_muts_[id]};

  bool is_initial_set = train_infos_[id]->train_pid == -1;
  if (is_initial_set && !pid.has_value()) {
    LOG(FATAL) << "initial set train info, pid must be valid";
  }

  if (is_initial_set) {
    LOG(INFO) << "initial set training info "
              << (pid.has_value() ? 
                  "pid " + std::to_string(pid.value()) : "")
              << (rank.has_value() ? 
                  " rank " + std::to_string(rank.value()) : "")
              << (world_size.has_value() ? 
                  " world_size " + std::to_string(world_size.value()) : "")
              << (init_batch_size.has_value() ? 
                  " init_batch_size " + std::to_string(init_batch_size.value()) : "")
              << (current_batch_size.has_value() ? 
                  " current_batch_size " + std::to_string(current_batch_size.value()) : "")
              << (model_name.has_value() ? 
                  " model_name " + std::string(model_name.value()) : "");
  }

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
    if (is_initial_set) {
      train_infos_[id]->target_batch_size = init_batch_size.value();
      train_infos_[id]->target_batch_size_unpublished = 
          init_batch_size.value();
    }
  }
  if (current_batch_size.has_value()) {
    train_infos_[id]->current_batch_size = current_batch_size.value();
  }
  if (model_name.has_value()) {
    strncpy(train_infos_[id]->model_name, model_name.value(), 256);
  }
}

std::vector<pid_t> InfTraSharedInfo::GetTrainPIDs() {
  std::vector<bip::scoped_lock<bip_mutex>> locks;
  locks.emplace_back(*train_info_muts_[0]);
  
  int train_world_size = train_infos_[0]->train_world_size;
  if (train_world_size == 0) {
    return {};
  }

  for (auto i : boost::irange(1, train_world_size)) {
    locks.emplace_back(*train_info_muts_[i]);
  }
  
  std::vector<pid_t> ret;
  for (auto i : boost::irange(0, train_world_size)) {
    ret.push_back(train_infos_[i]->train_pid);
  }
  return ret;
}

} // namespace ctrl
} // namespace colserve