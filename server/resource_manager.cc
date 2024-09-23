#include <server/logging_as_glog.h>
#include <server/resource_manager.h>
#include <server/train_launcher.h>
#include <server/train_adjuster.h>
#include <server/config.h>

#include <common/cuda_allocator.h>
#include <common/util.h>

#include <boost/algorithm/string.hpp>
#include <regex>


namespace colserve {

std::unique_ptr<ResourceManager> ResourceManager::resource_manager_;

ResourceManager::ResourceManager() {
  
}

double ResourceManager::GetFreeMemoryMB(int device_id, bool verbose) {
  using namespace sta;

  double free_memory_mb;
  double infer_memory_mb = GetInferMemoryMB(device_id);
  double train_memory_mb = GetTrainMemoryMB(device_id);
  double train_predict_memory_mb = 
      TrainAdjuster::PredictTrainMemUsageMB(device_id, verbose);

  if (Config::use_shared_tensor) {
    free_memory_mb = 
        sta::ByteToMB(sta::CUDAMemPool::Get(device_id)->PoolNbytes());
    free_memory_mb -= infer_memory_mb;
    free_memory_mb -= std::max(train_memory_mb, train_predict_memory_mb);
    free_memory_mb -= Config::train_memory_over_predict_mb;
  } else {
    auto [free, total] = Profiler::GetGPUMemInfo();
    free_memory_mb = ByteToMB(free);
    free_memory_mb = std::min(
        free_memory_mb, 
        sta::ByteToMB(total) 
        - infer_memory_mb 
        - std::max(train_predict_memory_mb, train_memory_mb) 
        - Config::train_memory_over_predict_mb
    );
  }

  LOG_IF(INFO, verbose && Config::log_memory_adjust) 
      << str(boost::format("[ResourceManager | Device %d]") % device_id)
      << " infer memory " << infer_memory_mb 
      << " train memory " << train_memory_mb 
      << " predict train memory " << train_predict_memory_mb
      << " free memory " << free_memory_mb;
            
  return free_memory_mb;
}

double ResourceManager::GetTrainAvailMemoryMB(int device_id, bool verbose) {
  using namespace sta;

  double infer_memory_mb = GetInferMemoryMB(device_id);

  double free_memory_mb;
  if (Config::use_shared_tensor) {
    free_memory_mb = 
        sta::ByteToMB(sta::CUDAMemPool::Get(device_id)->PoolNbytes());
    free_memory_mb -= infer_memory_mb;
    free_memory_mb -= Config::train_memory_over_predict_mb;
  } else {
    auto [free, total] = Profiler::GetGPUMemInfo();
    free_memory_mb = ByteToMB(free);
    free_memory_mb = std::min(
        free_memory_mb, 
        sta::ByteToMB(total) 
        - infer_memory_mb 
        - Config::train_memory_over_predict_mb
    );
  }

  LOG_IF(INFO, verbose) 
      << str(boost::format("[ResourceManager | Device %d]") % device_id)
      << " free memory " << free_memory_mb
      << " infer memory " << infer_memory_mb;

  return free_memory_mb;  
}

double ResourceManager::GetInferMemoryMB(int device_id) {
  using namespace sta;
  if (Config::use_shared_tensor_infer) {
    return ByteToMB(CUDAMemPool::Get(device_id)->InferMemUsage());
  } else {
    return ByteToMB(Profiler::GetLastInferMem(device_id));
  }
}

double ResourceManager::GetTrainMemoryMB(int device_id) {
  using namespace sta;
  if (Config::use_shared_tensor_train) {
    return ByteToMB(CUDAMemPool::Get(device_id)->TrainAllMemUsage());
  } else {
    return ByteToMB(Profiler::GetLastTrainMem(device_id));
  }
}

}