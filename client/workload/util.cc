#include "util.h"

namespace colserve {
namespace workload {

std::string ReadInput(const std::string &data_path) {
  std::ifstream data_file{data_path, std::ios::binary};
  CHECK(data_file.good()) << "data " << data_path << " not exist";
  std::string data{std::istreambuf_iterator<char>(data_file), std::istreambuf_iterator<char>()};
  data_file.close();
  return data;
}

AppBase::AppBase(const std::string &name) : app{name} {
  app.add_flag("--infer,!--no-infer", enable_infer, "enable infer workload");
  app.add_option("--infer-model", infer_models, "models of infer workload");
  app.add_flag("--train,!--no-train", enable_train, "enable train workload");
  app.add_option("--train-model", train_models, "models of train workload");
  app.add_option("-d,--duration", duration, "duration of workload");
  app.add_option("-c,--concurrency", concurrency, "concurrency of infer workload");
  app.add_option("--num-epoch", num_epoch, "num_epoch of train workload");
  app.add_option("--batch-size", batch_size, "batch_size of train workload");
  app.add_option_no_stream("-l,--log", log, "log file");
  app.add_option("-v,--verbose", verbose, "verbose level");
  app.add_option("--show-result", show_result, "show result");
  app.add_option( "-p,--port", port, "grpc port");
}




}
}
