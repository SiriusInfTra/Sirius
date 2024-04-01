#include "fake_engine.h" 
#include <torch/csrc/autograd/python_engine.h>
#include "torch_col/csrc/control_stub.h"

namespace torch_col {
Engine& GetTorchColEngine() {
  static FakeEngine *engine = new FakeEngine(python::PythonEngine::get_python_engine());
  return *engine;
}


static ColocateStub* colocate_stub_ = nullptr;

void SetUpTorchColEngine(ColocateStub *colocate_stub) {
  LOG(INFO) << "Register TorchCol Engine!";
  set_default_engine_stub(GetTorchColEngine);
  CHECK(colocate_stub_ == nullptr);
  colocate_stub_ = colocate_stub;
}


ColocateStub& GetColocateStub() {
  CHECK(colocate_stub_ != nullptr);
  return *colocate_stub_;
}
}  // namespace torch_col
