#ifndef COLSERVE_GRAPH_EXECUTOR
#define COLSERVE_GRAPH_EXECUTOR

// #include <tvm/runtime/device_api.h>
// #include <tvm/runtime/c_runtime_api.h>
#include <unordered_map>

#include <sta/tensor_methods.h>
#include <sta/tensor_pool.h>

#include "graph_executor_factory.h"


namespace colserve {
namespace tvm {

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                     \
  {                                         \
    int ret = (func);                       \
    CHECK_EQ(ret, 0) << TVMGetLastError();  \
  }


class GraphExecutor {
 public:
  GraphExecutor(GraphExecutorFactory &factory);
  void Init();
  void DeInit();
  void Run();
  void PipelineRun();
  const DLTensor* GetInput(int index) const;
  const DLTensor* GetInput(const std::string &index) const;
  const DLTensor* GetInputHostBuf(const std::string &index) const;
  const DLTensor* GetOutput(int index) const;
  const DLTensor* GetOutput(const std::string &index) const;
  const DLTensor* GetOutputHostBuf(const std::string &index) const;
  uint32_t GetInputIndex(const std::string &name) const;
  uint32_t GetOutputIndex(const std::string &name) const;

  TVMStreamHandle GetExecStream() const { return exec_stream_; }

  // void ResetBufStorage();
  // void ResetParamStorage();
  // void AllocBufStorage();
  // void AllocParamStorage();
  void ResetStorage();
  void AllocStorage();
  void LoadParams(bool pipeline = false);
  void ReSetupDataEntry();

  
  uint32_t GetNumOfNodes() const { return factory_.nodes_.size(); }
  uint32_t entry_id(NodeEntry e) const {
    return factory_.node_row_ptr_[e.node_id] + e.index;
  }
  uint32_t entry_id(uint32_t nid, uint32_t index) const {
    return factory_.node_row_ptr_[nid] + index;
  }
  

 private:
  void SetupStorage(bool alloc);
  void SetupOpExecs();
  std::pair<std::function<void()>, std::shared_ptr<OpArgs>> CreateTVMOp(
    const TVMOpParam &param, const std::vector<DLTensor*>& args);

  bool initialized_;
  GraphExecutorFactory &factory_;

  // std::vector<TVMArray> storage_pool_;
  // std::map<uint32_t, uint32_t> op_node_storage_id_map_;
  // std::vector<TVMArray> data_entry_;

  std::vector<sta::handle_t> storage_pool_;
  std::vector<sta::handle_t> data_entry_;

  // for no shared allocator
  std::vector<sta::STensor> raw_storage_pool_;
  std::vector<sta::STensor> raw_data_entry_;

  std::vector<size_t> data_alignment_;

  std::vector<std::function<void()>> op_execs_;
  // node input and output dltensors
  std::vector<std::vector<DLTensor*>> input_dltensors_;
  std::vector<std::vector<DLTensor*>> output_dltensors_;
  std::vector<std::vector<DLTensor*>> both_input_output_dltensors_;
  std::vector<std::vector<size_t>> input_param_nid_;

  // to avoid alloc pin memory during set input/get output
  std::unordered_map<std::string, TVMArray> input_cpu_pin_bufs_,
                                            output_cpu_pin_bufs_;

  // std::map<uint32_t, bool> param_ready_;
  std::vector<bool> param_ready_;

  void* blob_mem_{nullptr};
  
  TVMStreamHandle exec_stream_;
  TVMStreamHandle load_param_stream_;
};

}
}

#endif