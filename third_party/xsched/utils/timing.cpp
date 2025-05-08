#include "utils/log.h"
#include "utils/timing.h"

using namespace std::chrono;
using namespace xsched::utils;

Timer::~Timer()
{
    this->LogResults();
}

void Timer::RecordBegin()
{
    XASSERT(getrusage(RUSAGE_SELF, &usage_begin_) == 0, "getrusage failed");
    begin_ = system_clock::now();
}

void Timer::RecordEnd()
{
    XASSERT(getrusage(RUSAGE_SELF, &usage_end_) == 0, "getrusage failed");
    end_ = system_clock::now();
    total_ns_ += duration_cast<nanoseconds>(end_ - begin_).count();
    total_cpu_ns_ +=
        (GetCpuTimeUs(&usage_end_) - GetCpuTimeUs(&usage_begin_)) * 1000;
    total_cnt_++;
}

void Timer::LogResults()
{
    XINFO("[TIMER] %s cnt(%ld) avg %.2fus",
          kName.c_str(), total_cnt_, this->GetAvgNs() / 1000.0);
}

int64_t Timer::GetAvgNs()
{
    if (total_cnt_ == 0) return 0;
    return total_ns_ / total_cnt_;
}

int64_t Timer::GetAvgNsCpu()
{
    if (total_cnt_ == 0) return 0;
    return total_cpu_ns_ / total_cnt_;
}
