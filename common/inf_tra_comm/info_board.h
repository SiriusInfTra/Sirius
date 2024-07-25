#pragma once

#include <common/sm_partition.h>
#include <common/inf_tra_comm/bip_helper.h>
#include <common/util.h>

#include <optional>
#include <atomic>


namespace colserve {
namespace ctrl {

struct InferInfo {
  std::array<SMPartitioner::TpcData, MAX_DEVICE_NUM> tpc_datas;

};

struct TrainInfo {
  pid_t train_pid;
  int train_rank;
  int train_world_size;

  int init_batch_size;
  int current_batch_size;
};

class InfTraInfoBoard {
 public:
  InfTraInfoBoard(bool is_server, int train_world_size,
                  bip::managed_shared_memory &bip_shm,
                  bip::scoped_lock<bip_mutex> &lock);

  const InferInfo* GetInferInfo() {
    return infer_info_;
  }
  InferInfo* GetMutableInferInfo() {
    return infer_info_;
  }

  const TrainInfo* GetTrainInfo(int id) {
    CHECK(IsTrainInfoValid(id));
    return train_infos_[id];
  }
  TrainInfo* GetMutableTrainInfo(int id) {
    CHECK(IsTrainInfoValid(id));
    return train_infos_[id];
  }

  void InvalidateTrainInfo(int id) {
    bip::scoped_lock lock{*train_info_muts_[id]};
    train_infos_[id]->train_pid = -1;
  }

  bool IsTrainInfoValid(int id) {
    bip::scoped_lock lock{*train_info_muts_[id]};
    return train_infos_[id]->train_pid != -1;
  }

  void SetInferInfo(std::function<void(InferInfo*)> fn);
  void SetTrainInfo(int id, std::function<void(TrainInfo*)> fn);
  void SetTrainInfo(int id, std::optional<pid_t> pid, 
                    std::optional<int> rank, std::optional<int> world_size,
                    std::optional<int> init_batch_size,
                    std::optional<int> current_batch_size);

 private:
  std::string GetInferInfoName() {
    return "infer-info";
  }
  std::string GetTrainInfoName(int id) {
    return "train-info-" + std::to_string(id);
  }
  std::string GetInferInfoMutexName() {
    return "infer-info-mut";
  }
  std::string GetTrainInfoMutexName(int id) {
    return "train-info-mut-" + std::to_string(id);
  }

  InferInfo* infer_info_{nullptr};
  std::array<TrainInfo*, MAX_DEVICE_NUM> train_infos_{nullptr};

  bip_mutex *infer_info_mut_{nullptr};
  std::array<bip_mutex*, MAX_DEVICE_NUM> train_info_muts_{nullptr};

  bip::managed_shared_memory &bip_shm_;
};

}
}