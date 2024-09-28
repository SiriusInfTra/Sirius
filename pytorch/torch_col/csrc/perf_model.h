#pragma once

#include <torch_col/csrc/dist_train_sync.h>

namespace torch_col {

class PerfModel {
 public:
  static void Init();
  static void RecordThpt(int batch_size, double batch_time_ms);
  static double GetThpt(int batch_size);
  static std::vector<double> GetThptVec(const std::vector<int> &batch_sizes);

  PerfModel();  

 private:
  static std::unique_ptr<PerfModel> perf_model_;

  double GetThptWithLock(int batch_size);

  // batch size -> (thpt, count)
  bip_map<int, std::pair<double, int>> *batch_thpt_;
  bip_mutex *mut_;
};

}