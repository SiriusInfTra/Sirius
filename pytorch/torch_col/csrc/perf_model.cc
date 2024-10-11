#include <torch_col/csrc/perf_model.h>

namespace torch_col {

std::unique_ptr<PerfModel> PerfModel::perf_model_ = nullptr;

void PerfModel::Init() {
  CHECK(PerfModel::perf_model_ == nullptr)
      << "PerfModel has been initialized";
  PerfModel::perf_model_ = std::make_unique<PerfModel>();
  // LOG(INFO) << "PerfModel initialized";
}

PerfModel::PerfModel() {
  DistTrainSync::CreateCustomSharedData(
    "perf_model_shared_data",
    
    std::make_pair(std::string{"batch_thpt"}, 
        &batch_thpt_),
    std::make_pair(std::string{"mut"}, 
        &mut_)
  );
}

void PerfModel::RecordThpt(int batch_size, double batch_time_ms) {
  CHECK(perf_model_ != nullptr);

  double thpt =  static_cast<double>(batch_size) / batch_time_ms;
  bip::scoped_lock lock{*perf_model_->mut_};
  auto it = perf_model_->batch_thpt_->find(batch_size);
  if (it == perf_model_->batch_thpt_->end()) {
    perf_model_->batch_thpt_->emplace(
        batch_size, std::make_pair(thpt, 1));
  } else {
    auto &his_thpt = it->second;
    his_thpt.first = (his_thpt.first * his_thpt.second + thpt)
                     / (his_thpt.second + 1);
    his_thpt.second++;
  }
}

double PerfModel::GetThpt(int batch_size) {
  CHECK(perf_model_ != nullptr);

  if (batch_size <= 0) {
    return 0;
  }

  bip::scoped_lock lock{*perf_model_->mut_};
  return perf_model_->GetThptWithLock(batch_size);
}

std::vector<double> PerfModel::GetThptVec(
    const std::vector<int> &batch_sizes) {
  CHECK(perf_model_ != nullptr);

  std::vector<double> res(batch_sizes.size());
  bip::scoped_lock lock{*perf_model_->mut_};
  for (auto i : boost::irange(batch_sizes.size())) {
    res[i] = perf_model_->GetThptWithLock(batch_sizes[i]);
  }
  return res;
}

double PerfModel::GetThptWithLock(int batch_size) {
  if (batch_thpt_->empty()) {
    return -1;
  }

  if (batch_size <= 0) {
    return 0;
  }

  if (batch_thpt_->begin()->first >= batch_size) {
    return batch_thpt_->begin()->second.first / batch_thpt_->begin()->first * batch_size;
  }

  if (batch_thpt_->rbegin()->first <= batch_size) {
    return batch_thpt_->rbegin()->second.first / batch_thpt_->rbegin()->first * batch_size;
  }

  auto it = batch_thpt_->lower_bound(batch_size);
  assert(it != batch_thpt_->end());
  if (it->first != batch_size) {
    auto it_prev = it;
    it_prev--;
    // linear interpolation
    return (it->second.first - it_prev->second.first) 
           / (it->first - it_prev->first) * (batch_size - it_prev->first)
           + it_prev->second.first;
  } else {
    return it->second.first;
  }
}

} // namespace torch_col