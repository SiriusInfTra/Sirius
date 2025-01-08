#ifndef COLSERVE_SM_PARTITION_H
#define COLSERVE_SM_PARTITION_H

#include <common/util.h>

#include <atomic>
#include <thread>

namespace colserve {


class SMPartitioner {
 public:
  enum class Mode {
    // Infer can use all TPC, the number of Train TPC is total - infer
    kMaxUtilization, // currently only use this. 

    // Infer/Train use different TPCs
    kSeperateTpc,
  };

  struct TpcData {
    std::atomic<int> infer_required_tpc_num;
    uint64_t infer_tpc_mask; // not used 
  };

  static void Init(int device_id);
  static SMPartitioner* Get(int device_id);

  int GetGPUNumSM();
  int GetGPUNumTpc();
  void SetGlobalTpcMask(uint64_t mask_64);
  void SetStreamTpcMask(CUstream s, uint64_t mask_64);
  std::string CheckStreamSM(CUstream s);

  void SetInferRequiredTpcNum(int tpc_num);
  int GetInferRequiredTpcNum();
  void AddInferRequiredTpcNum(int tpc_num);
  void DecInferRequiredTpcNum(int tpc_num);
  uint64_t GetTrainAvailTpcMask();
  int GetTrainAvailTpcNum();

  uint64_t SetTrainStreamTpcMask(CUstream s);

  SMPartitioner(int device_id);
  ~SMPartitioner();

 private:
  static std::array<std::unique_ptr<SMPartitioner>, MAX_DEVICE_NUM> sm_partitioners_;

  int device_id_;
  int gpu_sm_num_;
  int gpu_tpc_num_;
  int sm_num_per_tpc_;

  // TpcData* tpc_data_;
  // std::atomic<int>* ref_cnt_;
  // std::string shm_name_;
  // bip::managed_shared_memory shm_;
  TpcData* tpc_data_;

  int min_train_tpc_num_ = 5;
#if 0 // v100
  int max_train_tpc_num_ = 40;
#else // a100
  int max_train_tpc_num_ = 56;
#endif

  // assume one thread per stream, so no need for lock  
  static thread_local 
  std::unordered_map<CUstream, uint64_t> stream_last_tpc_mask_map_;
};


}

#endif