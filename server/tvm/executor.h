#ifndef COLSERVE_GRAPH_EXECUTOR
#define COLSERVE_GRAPH_EXECUTOR

#include <server/tvm/graph.h>
#include <server/config.h>

#include <common/tensor/tensor_methods.h>
#include <common/tensor/tensor.h>
#include <common/cuda_allocator.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <memory>
#include <atomic>


namespace colserve {
namespace tvm {

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                     \
  {                                         \
    int ret = (func);                       \
    CHECK_EQ(ret, 0) << TVMGetLastError();  \
  }

class Executor {
 public:
  Executor(TVMGraph &tvm_graph, size_t worker_id, const std::vector<DLDevice> &devs);
  void Init(bool load_param);
  bool Initialized() const { return initialized_; }
  void FakeInit(bool malloc, bool load_param); // used for simulating unlimted get gpu resource
  void DeInit(const std::vector<size_t> &keep_cold_cached_group_id);
  void ClearColdCached(const std::vector<size_t> &cold_cached_group_id);

  void Run();
  void PipeLineLoad();
  void PipelineRun();
  const DLTensor* GetInput(int index) const;
  const DLTensor* GetInput(const std::string &index) const;
  const DLTensor* GetInputHostBuf(const std::string &index) const;
  const DLTensor* GetOutput(int index) const;
  const DLTensor* GetOutput(const std::string &index) const;
  const DLTensor* GetOutputHostBuf(const std::string &index) const;
  uint32_t GetInputIndex(const std::string &name) const;
  uint32_t GetOutputIndex(const std::string &name) const;

  TVMStreamHandle GetExecStream() const { 
    CHECK_EQ(initialized_, true);
    return exec_stream_;
  }

  // void ResetBufStorage();
  // void ResetParamStorage();
  // void AllocBufStorage();
  // void AllocParamStorage();
  void ResetStorage();
  void AllocStorage();

  // deprecated
  // void AllocStorageMaybeAdjust();

  void LoadParams(bool pipeline, bool force);
  void RefreshDataEntry();

  
  uint32_t GetNumOfNodes() const { return tvm_graph_.nodes_.size(); }
  // uint32_t entry_id(NodeEntry e) const {
  //   return tvm_graph_.node_row_ptr_[e.node_id] + e.index;
  // }
  // uint32_t entry_id(uint32_t nid, uint32_t index) const {
  //   return tvm_graph_.node_row_ptr_[nid] + index;
  // }

  // size_t GetParamStorageSize() const {
  //   return param_storage_size_;
  // }

  // size_t GetBufferStorageSize() const {
  //   return buffer_storage_size_;
  // }

  // size_t GetStorageSize() const {
  //   return param_storage_size_ + buffer_storage_size_;
  // }

  // size_t GetStorageSizeAlign() const {
  //   if (Config::group_param_load) {
  //     if (Config::group_param_nbytes_with_fragment) {
  //       return model_nbytes_with_group_fragment_;
  //     } else {
  //       return AlignedNBytes<ALIGN_NBYTES>(GetStorageSize());
  //     }
  //   } else {
  //     return GetStorageSize();
  //   }
  // }

  size_t GetMissingStorageSizeAlign() const;

  // std::vector<size_t> GetGroupsNbytes() const {
  //   return storage_group_nbytes_;
  // }


 private:
  void SetupStorage(bool alloc);
  // void SetupStorageGroup();
  void SetupHostPinnedIOStorage();
  void LoadParamGroupParti(const std::string &path);
  void SetupOpExecs();
  std::pair<std::function<void()>, std::shared_ptr<OpArgs>> CreateTVMOp(
    const TVMOpParam &param, const std::vector<DLTensor*>& args);

  bool initialized_;
  size_t infer_model_worker_id_;
  TVMGraph &tvm_graph_;

  std::vector<DLDevice> devices_;


  // std::vector<TVMArray> storage_pool_;
  // std::map<uint32_t, uint32_t> op_node_storage_id_map_;
  // std::vector<TVMArray> data_entry_;

  // std::vector<PoolEntry> pool_entry_;

  std::vector<sta::STensor> storage_pool_;
  std::vector<sta::STensor> data_entry_;
  // if use storage group, storage pool will be view of grouped storage
  std::vector<std::shared_ptr<sta::CUDAMemPool::PoolEntry>> storage_group_;

  // for no shared allocator
  // std::vector<sta::STensor> raw_storage_pool_;
  // std::vector<sta::STensor> raw_data_entry_;

  // std::vector<size_t> data_alignment_;

  std::vector<std::function<void()>> op_execs_;
  // node input and output dltensors
  // std::vector<std::vector<DLTensor*>> input_dltensors_;
  // std::vector<std::vector<DLTensor*>> output_dltensors_;
  // std::vector<std::vector<DLTensor*>> both_input_output_dltensors_;
  std::vector<std::vector<size_t>> input_param_eid_; // node id -> [param data entry ids]


  // to avoid alloc pin memory during set input/get output
  std::unordered_map<std::string, TVMArray> 
      input_host_pin_bufs_, output_host_pin_bufs_;

  // std::map<uint32_t, bool> param_ready_;
  std::vector<std::unique_ptr<std::atomic<bool>>> param_ready_;
  std::vector<cudaEvent_t> param_ready_events_; // for pipeline
  std::vector<uint32_t> param_ready_event_ids_; // for group param pipeline
  // std::vector<cudaEvent_t> pipeline_op_exec_starts_, pipeline_op_exec_ends_;

  // pipeline load params and exec
  std::future<void> load_params_future_;
  std::mutex pipeline_load_params_mut_;
  std::condition_variable pipeline_load_params_cv_;

  // void* blob_mem_{nullptr};
  std::shared_ptr<sta::CUDAMemPool::PoolEntry> blob_mem_{nullptr};

  // group storage
  // std::vector<uint32_t> storage_alloc_order_;

  // better alloc to avoid fragmentation
  // size_t model_nbytes_with_group_fragment_;
  // std::vector<size_t> storage_group_nbytes_;
  // std::vector<std::shared_ptr<sta::CUDAMemPool::PoolEntry>> storage_group_;

  // cached group, used for SetupMemory/Init(false)
  std::unordered_map<size_t, std::shared_ptr<sta::CUDAMemPool::PoolEntry>> cold_cached_group_;
  std::atomic<size_t> cold_cached_nbytes_;

  // [ param storage group, [param ids ...] ]
  std::vector<std::pair<TVMArray, std::vector<uint32_t>>> host_param_storage_group_;

  // std::vector<size_t> storage_group_parti_;

  
  TVMStreamHandle exec_stream_;
  TVMStreamHandle load_param_stream_;


  // size_t param_storage_size_ = 0;
  // size_t buffer_storage_size_ = 0;
};

}
}

#endif