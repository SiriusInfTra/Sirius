#ifndef COLSERVE_SM_PARTITION_H
#define COLSERVE_SM_PARTITION_H

#include <common/util.h>
#include <boost/interprocess/managed_shared_memory.hpp>

#include <thread>

namespace colserve {

namespace bip = boost::interprocess;

class SMPartitioner {
 public:
  enum class Mode {
    // Infer can use all TPC, the number of Train TPC is total - infer
    kMaxUtilization, // currently only use this. 

    // Infer/Train use different TPCs
    kSeperateTpc,
  };

  static void Init(int device, bool cleanup, bool observe);
  static int GetGPUNumSM();
  static int GetGPUNumTpc();
  static void SetGlobalTpcMask(uint64_t mask_64);
  static void SetStreamTpcMask(CUstream s, uint64_t mask_64);
  static std::string CheckStreamSM(CUstream s);

  static void SetInferRequiredTpcNum(int tpc_num);
  static int GetInferRequiredTpcNum();
  static void AddInferRequiredTpcNum(int tpc_num);
  static void DecInferRequiredTpcNum(int tpc_num);
  static uint64_t GetTrainAvailTpcMask();
  static int GetTrainAvailTpcNum();

  static uint64_t SetTrainStreamTpcMask(CUstream s);

  SMPartitioner(int device, bool cleanup, bool observe);
  ~SMPartitioner();

 private:
  static std::unique_ptr<SMPartitioner> sm_partitioner_;

  struct TpcData {
    std::atomic<int> infer_required_tpc_num;
    uint64_t infer_tpc_mask; // not used 
  };

  int device_;
  int gpu_sm_num_;
  int gpu_tpc_num_;
  int sm_num_per_tpc_;

  TpcData* tpc_data_;
  std::atomic<int>* ref_cnt_;
  std::string shm_name_;
  bip::managed_shared_memory shm_;

  int min_train_tpc_num_ = 1;

  // assume one thread per stream, so no need for lock  
  static thread_local std::unordered_map<CUstream, uint64_t> stream_last_tpc_mask_map_;
};


}

#endif