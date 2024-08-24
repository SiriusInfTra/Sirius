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
  pid_t train_pid{-1};
  int train_rank{0};
  int train_world_size{0};

  int init_batch_size{0};
  int current_batch_size{0};

  int target_batch_size{0};
  int target_batch_size_unpublished{0};

  char model_name[256];
};

class InfTraSharedInfo {
 public:
  InfTraSharedInfo(bool is_server, int train_world_size,
                  bip::managed_shared_memory &bip_shm,
                  bip::scoped_lock<bip_mutex> &lock);

  const InferInfo* GetInferInfoUnsafe() {
    return infer_info_;
  }
  InferInfo* GetMutableInferInfoUnsafe() {
    return infer_info_;
  }

  const TrainInfo* GetTrainInfoUnsafe(int id) {
    CHECK(IsTrainInfoValid(id));
    return train_infos_[id];
  }
  TrainInfo* GetMutableTrainInfoUnsafe(int id) {
    CHECK(IsTrainInfoValid(id));
    return train_infos_[id];
  }
  std::vector<pid_t> GetTrainPIDs();

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
                    std::optional<int> rank, 
                    std::optional<int> world_size,
                    std::optional<int> init_batch_size,
                    std::optional<int> current_batch_size,
                    std::optional<const char*> model_name);

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