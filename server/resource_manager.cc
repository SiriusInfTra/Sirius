#include <common/cuda_allocator.h>
#include <common/util.h>

#include <server/resource_manager.h>
#include <server/train_launcher.h>
#include <server/config.h>


namespace colserve {

std::unique_ptr<ResourceManager> ResourceManager::resource_manager_;

double ResourceManager::GetFreeMemoryMB() {
  using namespace sta;

  double free_memory_mb;
  double infer_memory_mb = GetInferMemoryMB();
  double train_memory_mb = GetTrainMemoryMB();
  double train_predict_memory_mb = TrainLauncher::Get()->PredictMemUsageMB();

  if (Config::use_shared_tensor) {
    free_memory_mb = sta::ByteToMB(sta::CUDAMemPool::PoolNbytes());
  } else {
    auto [free, total] = Profiler::GetGPUMemInfo();
    free_memory_mb = ByteToMB(free);
  }
  free_memory_mb -= infer_memory_mb;
  free_memory_mb -= std::max(train_memory_mb, train_predict_memory_mb);
  free_memory_mb -= Config::train_memory_over_predict_mb;

  LOG(INFO) << "[ResourceManager] "
            << " infer memory " << infer_memory_mb 
            << " train memory " << train_memory_mb 
            << " predict train memory " << train_predict_memory_mb
            << " free memory " << free_memory_mb;
            
  return free_memory_mb;
}

double ResourceManager::GetTrainAvailMemoryMB() {
  using namespace sta;

  double infer_memory_mb = GetInferMemoryMB();

  double free_memory_mb;
  if (Config::use_shared_tensor) {
    free_memory_mb = sta::ByteToMB(sta::CUDAMemPool::PoolNbytes());
  } else {
    auto [free, total] = Profiler::GetGPUMemInfo();
    free_memory_mb = ByteToMB(free);
  }

  free_memory_mb -= infer_memory_mb;
  free_memory_mb -= Config::train_memory_over_predict_mb;

  LOG(INFO) << "[ResourceManager] "
            << " free memory " << free_memory_mb
            << " infer memory " << infer_memory_mb;

  return free_memory_mb;  
}

}