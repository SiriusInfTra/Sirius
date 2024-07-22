#pragma once

#include <common/inf_tra_comm/bip_helper.h>
#include <common/util.h>

#include <atomic>


namespace colserve {
namespace ctrl {

struct InferInfo {
  std::atomic<int> required_tpc_num;

};

struct TrainInfo {
  int batch_size;

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
    return train_infos_[id];
  }
  TrainInfo* GetMutableTrainInfo(int id) {
    return train_infos_[id];
  }

 private:
  std::string GetInferInfoName() {
    return "infer-info";
  }
  std::string GetTrainInfoName(int id) {
    return "train-info-" + std::to_string(id);
  }

  InferInfo* infer_info_{nullptr};
  std::array<TrainInfo*, MAX_DEVICE_NUM> train_infos_{nullptr};

  bip::managed_shared_memory &bip_shm_;
};

}
}