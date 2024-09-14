#pragma once

#include <common/log_as_glog_sta.h>
#include <common/sm_partition.h>
#include <common/inf_tra_comm/bip_helper.h>
#include <common/util.h>

#include <boost/range/irange.hpp>
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

  template <typename ValueType>
  ValueType GetTrainInfoField(int id, size_t field_off) {
    bip::scoped_lock lock{*train_info_muts_[id]};
    return GetTrainInfoFieldWithoutLock<ValueType>(id, field_off);
  }

  template <typename ValueType>
  std::vector<ValueType> GetTrainInfoFieldVec(size_t field_off) {
    std::vector<bip::scoped_lock<bip_mutex>> locks;
    locks.emplace_back(*train_info_muts_[0]);

    int train_world_size = train_infos_[0]->train_world_size;
    if (train_world_size == 0) {
      return {};
    }

    for (int i : boost::irange(1, train_world_size)) {
      locks.emplace_back(*train_info_muts_[i]);
    }

    std::vector<ValueType> ret;
    for (int i : boost::irange(train_world_size)) {
      ret.push_back(
          GetTrainInfoFieldWithoutLock<ValueType>(i, field_off));
    }
    return ret;
  }

  template <typename ValueType>
  void UpdateTrainInfoFieldVec(size_t field_off,
                               const std::vector<ValueType> &values) {
    std::vector<bip::scoped_lock<bip_mutex>> locks;
    locks.emplace_back(*train_info_muts_[0]);
    int train_world_size = train_infos_[0]->train_world_size;
    if (train_world_size == 0) {
      return;
    }
    
    CHECK_EQ(values.size(), train_world_size);
    for (int i : boost::irange(1, train_world_size)) {
      locks.emplace_back(*train_info_muts_[i]);
    }

    for (int i : boost::irange(train_world_size)) {
      UpdateTrainInfoFieldWithoutLock<ValueType>(
          i, field_off, values[i]);
    }
  }


  void InvalidateTrainInfo(int id) {
    bip::scoped_lock lock{*train_info_muts_[id]};
    train_infos_[id]->train_pid = -1;
  }

  bool IsTrainInfoValid(int id) {
    bip::scoped_lock lock{*train_info_muts_[id]};
    return IsTrainInfoValidWithoutLock(id);
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

  template <typename ValueType>
  ValueType GetTrainInfoFieldWithoutLock(int id, size_t field_off) {
    CHECK(IsTrainInfoValidWithoutLock(id));
    return *reinterpret_cast<ValueType*>(
        reinterpret_cast<char*>(train_infos_[id]) + field_off);
  }

  template <typename ValueType>
  void UpdateTrainInfoFieldWithoutLock(int id, size_t field_off, 
                                       ValueType value) {
    CHECK(IsTrainInfoValidWithoutLock(id));
    ValueType *field_ptr = reinterpret_cast<ValueType*>(
        reinterpret_cast<char*>(train_infos_[id]) + field_off);
    *field_ptr = value;
  }

  inline bool IsTrainInfoValidWithoutLock(int id) {
    return train_infos_[id]->train_pid != -1;
  }

  InferInfo* infer_info_{nullptr};
  std::array<TrainInfo*, MAX_DEVICE_NUM> train_infos_{nullptr};

  bip_mutex *infer_info_mut_{nullptr};
  std::array<bip_mutex*, MAX_DEVICE_NUM> train_info_muts_{nullptr};

  bip::managed_shared_memory &bip_shm_;
};


#define COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD(id, field) \
  ::colserve::ctrl::InfTraCommunicator:: \
  GetSinfo()->GetTrainInfoField< \
    decltype(::colserve::ctrl::TrainInfo::field) \
  >(id, offsetof(::colserve::ctrl::TrainInfo, field))


#define COMMUNICATOR_GET_SHARED_TRAIN_INFO_FIELD_VEC(field) \
  ::colserve::ctrl::InfTraCommunicator:: \
  GetSinfo()->GetTrainInfoFieldVec< \
    decltype(::colserve::ctrl::TrainInfo::field) \
  >(offsetof(::colserve::ctrl::TrainInfo, field))


#define COMMUNICATOR_UPDATE_SHARED_TRAIN_INFO_FIELD_VEC(field, values) \
  ::colserve::ctrl::InfTraCommunicator:: \
  GetSinfo()->UpdateTrainInfoFieldVec< \
    decltype(::colserve::ctrl::TrainInfo::field) \
  >(offsetof(::colserve::ctrl::TrainInfo, field), values)

}
}