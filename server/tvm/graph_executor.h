#ifndef COLSERVE_GRAPH_EXECUTOR
#define COLSERVE_GRAPH_EXECUTOR

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>

#include "graph_executor_factory.h"


namespace colserve {
namespace tvm {

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                     \
  {                                         \
    int ret = (func);                       \
    CHECK_EQ(ret, 0) << TVMGetLastError(); \
  }


class GraphExecutor {
 public:
  GraphExecutor(GraphExecutorFactory &factory);
  void Init();
  void DeInit();
  void Run();
  void PipelineRun();
  TVMArray GetInput(int index) const;
  TVMArray GetInput(const std::string &index) const;
  TVMArray GetOutput(int index) const;
  TVMArray GetOutput(const std::string &index) const;
  uint32_t GetInputIndex(const std::string &name) const;
  uint32_t GetOutputIndex(const std::string &name) const;

  // void ResetBufStorage();
  // void ResetParamStorage();
  // void AllocBufStorage();
  // void AllocParamStorage();
  void ResetStorage();
  void AllocStorage();
  void LoadParams();
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

  std::vector<TVMArray> storage_pool_;
  // std::map<uint32_t, uint32_t> op_node_storage_id_map_;
  std::vector<TVMArray> data_entry_;
  std::vector<size_t> data_alignment_;

  std::vector<std::function<void()>> op_execs_;
  std::vector<std::vector<DLTensor*>> input_dltensors_;
  std::vector<std::vector<DLTensor*>> output_dltensors_;
  std::vector<std::vector<DLTensor*>> both_input_output_dltensors_;
  std::vector<std::vector<size_t>> input_param_nid_;

  // std::map<uint32_t, bool> param_ready_;
  std::vector<bool> param_ready_;

  
  TVMStreamHandle run_stream_;
  TVMStreamHandle load_param_stream_;
};

}
}

#endif