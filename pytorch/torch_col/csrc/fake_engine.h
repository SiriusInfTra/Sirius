#ifndef TORCH_COL_FAKE_ENGINE_H
#define TORCH_COL_FAKE_ENGINE_H
#include "mem_tagging.h"
#include "control_stub.h"
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/python_engine.h>
#include <memory>


namespace torch_col {
using namespace torch::autograd;

class FakeEngine: public Engine {
 private:
  Engine &python_engine_;
 public:
  FakeEngine(Engine &python_engine) : python_engine_(python_engine) {}
  virtual variable_list execute(
      const edge_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      bool create_graph,
      bool accumulate_grad,
      const edge_list& outputs = {}) {
    return python_engine_.execute(roots, inputs, keep_graph, create_graph, accumulate_grad, outputs);
  }

  virtual c10::intrusive_ptr<at::ivalue::Future> execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  return python_engine_.execute_with_graph_task(graph_task, graph_root, std::move(input_buffer));
  }

  virtual std::unique_ptr<AnomalyMetadata> make_anomaly_metadata() {
    return python_engine_.make_anomaly_metadata();
  }

  virtual std::unique_ptr<SavedVariableHooks> get_default_saved_variable_hooks() {
    return std::make_unique<TorchColSavedVariableHooks>();
  }

  virtual void thread_on_exception(
    std::shared_ptr<GraphTask> graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
    return python_engine_.thread_on_exception(graph_task, fn, e);
  }

  virtual void thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment = true) {
    python_engine_.thread_init(device, ready_queue, should_increment);
  }

};


Engine& GetTorchColEngine();

void SetUpTorchColEngine();

void SetUpTorchColEngine(ColocateStub *colocate_stub);

ColocateStub& GetColocateStub();
}
#endif
