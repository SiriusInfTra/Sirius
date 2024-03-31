#include "fake_engine.h" 
#include <torch/csrc/autograd/python_engine.h>

namespace torch_col {
Engine& GetTorchColEngine() {
  static FakeEngine *engine = new FakeEngine(python::PythonEngine::get_python_engine());
  return *engine;
}

void SetUpTorchColEngine() {
  LOG(INFO) << "Register TorchCol Engine";
  set_default_engine_stub(GetTorchColEngine);
}


static ColocateStub* colocate_stub = nullptr;

void SetupColocateStub(size_t batch_size) {
  CHECK(colocate_stub == nullptr);
  colocate_stub = new ColocateStub(batch_size);
}

ColocateStub& GetColocateStub() {
  CHECK(colocate_stub != nullptr);
  return *colocate_stub;
}
}  // namespace torch_col
