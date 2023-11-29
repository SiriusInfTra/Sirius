#include "util.h"
#include "glog/logging.h"

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
  app.add_option("--delay-before-infer", delay_before_infer, "delay before start infer.");


  app.add_option("--seed", seed, "random seed");

  app.add_flag("--warm-up", warmup, "warm up infer model");
}

uint64_t AppBase::seed = static_cast<uint64_t>(-1);

std::vector<std::vector<unsigned>> load_azure(unsigned trace_id,
                                              unsigned period) {
  const char path_template[] =
      "./workload_data/azurefunctions-dataset2019/"
      "function_durations_percentiles.anon.d%02d.csv";
  char ch[sizeof(path_template)];
  std::string line;
  std::vector<std::pair<unsigned, std::vector<unsigned>>> tmp;
  std::vector<std::vector<unsigned>> result;
  CHECK_GE(trace_id, 1);
  CHECK_LE(trace_id, 14);
  sprintf(ch, path_template, trace_id);
  LOG(INFO) << "Open trace file: " << ch << ".";
  std::ifstream f{ch};
  CHECK(f.is_open());
  std::getline(f, line);  // skip headers
  while (std::getline(f, line)) {
    std::istringstream ss{line};
    std::string token;
    std::vector<unsigned> sizes(std::min(1440U, period));
    for (size_t k = 0; k < 4; ++k) {  // skip first 4 token
      std::getline(ss, token, ',');
    }
    for (auto& i : sizes) {
      std::getline(ss, token, ',');
      i = std::stoi(token);
    }
    unsigned sum = std::accumulate(sizes.cbegin(), sizes.cend(), 0U);
    tmp.emplace_back(sum, std::move(sizes));
  }
  std::sort(tmp.begin(), tmp.end(),
            [](auto&& x, auto&& y) { return x.first < y.first; });
  for (auto&& size : tmp) {
    result.emplace_back(std::move(size.second));
  }
  return result;
}
CountDownLatch::CountDownLatch(int count) : count_(count) {}


void CountDownLatch::CountDownAndWait() {
  std::unique_lock lock{mutex_};
  if (--count_ == 0) {
    condition_.notify_all();
  }
  condition_.wait(lock, [&] { return count_ == 0; });
}


void CountDownLatch::Reset(int count) {
  std::unique_lock lock{mutex_};
  count_ = count;
  if (count_ == 0) {
    condition_.notify_all();
  }
}

void CountDownLatch::Wait() {
  std::unique_lock lock{mutex_};
  condition_.wait(lock, [&] { return count_ == 0; });
}
}  // namespace workload
}
