#pragma once

#include <torch_col/csrc/config.h>

#include <common/inf_tra_comm/bip_helper.h>
#include <common/inf_tra_comm/message_queue.h>

#include <pthread.h>
#include <type_traits>
#include <experimental/type_traits>
#include <utility>

namespace torch_col {

using namespace colserve;

class DistTrainSync {
 public:
  static void Init();
  static bool IsInitialized();
  static void WaitBarrier();
  static void Send(int dst, const std::string &msg);
  static std::string Recv(int src);

  DistTrainSync();
  ~DistTrainSync();

  // all training must call this function to create message queue
  template<typename T>
  static bool CreateCustomMq(const std::string &name) {
    CHECK(dist_train_sync_ != nullptr);

    if (dist_train_sync_->intra_train_custom_mq_dict_.find(name) != 
        dist_train_sync_->intra_train_custom_mq_dict_.end()) {
      return false;
    }

    auto mq = new array_2d_t<void*, MAX_DEVICE_NUM, MAX_DEVICE_NUM>{nullptr};
    auto atomic_func = [&]() {
      for (int i = 0; i < MAX_DEVICE_NUM; ++i) {
        for (int j = 0; j < MAX_DEVICE_NUM; ++j) {
          mq->at(i).at(j) = new ctrl::BasicMessageQueue<T>(
            TorchColConfig::IsTrainMaster(), 
            dist_train_sync_->GetMqName(name, i, j),
            dist_train_sync_->bip_shm_);
        }
      }
    };
    
    if (TorchColConfig::IsTrainMaster()) {
      atomic_func();
      dist_train_sync_->WaitBarrier();
    } else {
      dist_train_sync_->WaitBarrier();
      atomic_func();
    }

    dist_train_sync_->intra_train_custom_mq_dict_[name] = mq;
    return true;
  }

  template<typename T>
  static ctrl::BasicMessageQueue<T>* GetCustomMq(
      const std::string &name, int src, int dst) {
    CHECK(dist_train_sync_ != nullptr);
    CHECK_LE(src, MAX_DEVICE_NUM);
    CHECK_LE(dst, MAX_DEVICE_NUM);

    auto it = dist_train_sync_->intra_train_custom_mq_dict_.find(name);
    if (it == dist_train_sync_->intra_train_custom_mq_dict_.end()) {
      return nullptr;
    }
    
    auto mqs = it->second;
    return static_cast<ctrl::BasicMessageQueue<T>*>(
        mqs->at(src).at(dst));
  }

  template<typename ... Args>
  static bool CreateCustomSharedData(
      const std::string &prefix_name,
      std::pair<std::string, Args**> ... data_list) {
    CHECK(dist_train_sync_ != nullptr);

    // check if the data already exists
    auto check_name_conflict = [&](const std::string & name_) -> bool {
      auto name = prefix_name + '-' + name_;
      if (dist_train_sync_->intra_train_custom_shared_data_dict_.find(name) 
          != dist_train_sync_->intra_train_custom_shared_data_dict_.end()) {
        return true;
      }
      return false;
    };

    bool name_conflict = (check_name_conflict(data_list.first) || ...);
    if (name_conflict) {
      return false;
    }

    auto find_or_create_fn = [&](const std::string& name_, auto** data) {
      auto name = prefix_name + '-' + name_;
      using data_p_t = typename std::remove_pointer_t<decltype(data)>;
      using data_t = typename std::remove_pointer_t<data_p_t>;
      if constexpr (colserve::is_bip_container<data_t>::value) {
        if (TorchColConfig::IsTrainMaster()) {
          *data = dist_train_sync_->bip_shm_.find_or_construct<data_t>
              (name.c_str())
              (dist_train_sync_->bip_shm_.get_segment_manager());
        } else {
          *data = dist_train_sync_->bip_shm_.find<data_t>(name.c_str()).first;
        }
      }  else {
        static_assert(
          !std::experimental::is_detected_v<colserve::cont_allocator_t, data_t>, 
          "data type is not supported");
        if (TorchColConfig::IsTrainMaster()) {
          *data = dist_train_sync_->bip_shm_.find_or_construct<data_t>
              (name.c_str())();
        } else {
          *data = dist_train_sync_->bip_shm_.find<data_t>(name.c_str()).first;
        }
      }
      dist_train_sync_->intra_train_custom_shared_data_dict_[name] = *data;
    };

    auto atomic_func = [&]() {
      (find_or_create_fn(data_list.first, data_list.second), ...);
    };

    if (TorchColConfig::IsTrainMaster()) {
      atomic_func();
      dist_train_sync_->WaitBarrier();
    } else {
      dist_train_sync_->WaitBarrier();
      atomic_func();
    }

    return true;
  }

 private:
  static std::unique_ptr<DistTrainSync> dist_train_sync_;
  static bool initialized_;

  constexpr static size_t kMsgBufSize = 256;
  using dist_train_str_mq_t = 
      colserve::ctrl::BasicMessageQueue<std::array<char, kMsgBufSize>>;

  std::string GetMqName(const std::string &name, int src, int dst) {
    return str(boost::format("dist-train-%s-mq-%d-%d") % name % src % dst);
  }

  pthread_barrier_t *barrier_{nullptr};

  std::unique_ptr<dist_train_str_mq_t> 
    intra_train_str_mqs_[colserve::MAX_DEVICE_NUM]
                        [colserve::MAX_DEVICE_NUM] = {nullptr};

  std::unordered_map<
      std::string, 
      array_2d_t<void*, MAX_DEVICE_NUM, MAX_DEVICE_NUM> *
  > intra_train_custom_mq_dict_;

  std::unordered_map<std::string, void*> 
      intra_train_custom_shared_data_dict_;

  std::string shm_name_;
  colserve::bip::managed_shared_memory bip_shm_;
};


}