#include <common/log_as_glog_sta.h>
#include <torch_col/csrc/fake_engine.h>
#include <torch_col/csrc/control_stub.h>

#include <torch/csrc/autograd/python_engine.h>

namespace torch_col {

static StubBase* stub_ptr_ = nullptr;

Engine& GetTorchColEngine() {
  static FakeEngine *engine = new FakeEngine(python::PythonEngine::get_python_engine());
  return *engine;
}

void SetUpTorchColEngine(StubBase *stub_ptr) {
  LOG(INFO) << "Register TorchCol Engine!";
  set_default_engine_stub(GetTorchColEngine);
  CHECK(stub_ptr_ == nullptr);
  stub_ptr_ = stub_ptr;
}

StubBase& GetColocateStub() {
  CHECK(stub_ptr_ != nullptr);
  return *stub_ptr_;
}
}  // namespace torch_col
