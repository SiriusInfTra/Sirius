#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <common/util.h>

#include <boost/lockfree/policies.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/circular_buffer.hpp>

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <thread>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <glog/logging.h>

namespace colserve::sta {

const static constexpr size_t MEM_BLOCK_NBYTES = 32_MB; /* 32M */

namespace detail {
constexpr size_t alignment = 1024;
constexpr size_t train_alloc_threshold = 256_MB;
constexpr size_t train_alloc_threshold_small = 32_MB;

const constexpr size_t MIN_BLOCK_NBYTES = 512; /* 512B */
const constexpr size_t SMALL_BLOCK_NBYTES = 1_MB;  /* 1MB */
const constexpr size_t SMALL_PAGE_NBYTES  = 2_MB; /* 2MB  */
const constexpr size_t LARGE_PAGE_NBYTES  = 32_MB; /* 32MB */

inline size_t GetAlignedNbytes(size_t nbytes) {
  static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
  return (nbytes + (alignment - 1)) & (~(alignment - 1));
}
inline size_t GetAlignedNbytes(size_t nbytes, size_t alignment_) {
  assert((alignment_ & (alignment_ - 1)) == 0);
  return (nbytes + (alignment_ - 1)) & (~(alignment_ - 1));
}
inline double ByteToMB(size_t nbytes) {
  return static_cast<double>(nbytes) / 1_MB;
}

inline std::string ByteDisplay(size_t nbytes) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2)
     << static_cast<double>(nbytes) / 1_MB << "MB (" << nbytes << " Bytes)";
  return ss.str();
}

template<size_t align>
inline size_t AlignedNBytes(size_t nbytes) {
  static_assert((align & (align - 1)) == 0, "alignment must be power of 2");
  return (nbytes + (align - 1)) & (~(align - 1));
}

inline size_t RoundUpPower2NBytes(size_t nbytes) {
  static std::vector<size_t> round_nbytes = []{
    std::vector<size_t> tmp;
    size_t start_nbytes = 1_MB; /* 1MB */
    size_t end_nbytes = 16_GB; /* 16GB */
    size_t roundup_power2_divisions = 4;
    for (size_t curr_nbytes = start_nbytes; curr_nbytes <= end_nbytes; curr_nbytes *= 2) {
      for (size_t k=0; k<roundup_power2_divisions; k++) {
        size_t nbytes = curr_nbytes + curr_nbytes / roundup_power2_divisions * k;
        if (nbytes % SMALL_PAGE_NBYTES == 0 || true) {
          tmp.push_back(nbytes);
        }

      }
    }
    return tmp;
  }();         
  return *std::lower_bound(round_nbytes.cbegin(), round_nbytes.cend(), nbytes);
}

inline std::pair<bool, size_t> AlignNbytes(size_t nbytes) {
  // return std::make_pair(false, RoundUpPower2NBytes(nbytes));
  if (nbytes <= SMALL_BLOCK_NBYTES - MIN_BLOCK_NBYTES) {
    /* small block even align to n*MIN_BLOCK_NBYTES */
    return std::make_pair(true, AlignedNBytes<MIN_BLOCK_NBYTES>(nbytes));
  } else {
    return std::make_pair(false, AlignedNBytes<MIN_BLOCK_NBYTES>(nbytes));
  }
}
} // namespace detail


enum class Belong {
  kTrain, kInfer, kFree
};

struct PhyMem {
  const size_t index;
  CUmemGenericAllocationHandle cu_handle;
  Belong * const belong;
};
namespace bip = boost::interprocess;



class HandleTransfer {
  // typically, there is a limit on the maximum number of transferred FD
  // include/net/scm.h SCM_MAX_FD 253
  static const constexpr size_t TRANSFER_CHUNK_SIZE = 128;
private:
  bip::managed_shared_memory  &shared_memory_;
  bip::interprocess_mutex     *request_mutex_;
  bip::interprocess_condition *request_cond_;
  bip::interprocess_mutex     *ready_mutex_;
  bip::interprocess_condition *ready_cond_;

  std::string         master_name_;
  std::string         slave_name_;
  std::vector<PhyMem> &phy_mem_list_;
  size_t              mem_block_nbytes_;

  /* master only */
  std::atomic<bool>            vmm_export_running_;
  std::unique_ptr<std::thread> vmm_export_thread_;

  void SendHandles(int fd_list[], size_t len, bip::scoped_lock<bip::interprocess_mutex> &ready_lock);

  void ReceiveHandle(int fd_list[], size_t len);

  void ExportWorker();

public:
  HandleTransfer(bip::managed_shared_memory &shm, std::vector<PhyMem> &phy_mem_list, size_t mem_block_nbytes, size_t mem_block_num);

  void InitMaster();

  void InitSlave();

  void ReleaseMaster();
};

class MemPool {
  using ring_buffer = boost::circular_buffer<size_t, bip::allocator<size_t, bip::managed_shared_memory::segment_manager>>;
private:
  static std::unique_ptr<MemPool> instance_;

  bip::managed_shared_memory  shared_memory_;
  bip::interprocess_mutex     *mutex_;
  int                         *ref_count_;
  std::string                 shared_memory_name_;

  std::unique_ptr<HandleTransfer> tranfer;
  std::vector<PhyMem> phy_mem_list_;
  ring_buffer *free_queue;
  
  bool is_master_;
public:
  static MemPool &Get();

  const size_t mempool_nbytes;

  MemPool(size_t nbytes, bool cleanup);

  void WaitSlaveExit();

  size_t AllocPhyMem(std::vector<PhyMem *> &phy_mem_list, Belong belong);

  void DeallocPhyMem(const std::vector<PhyMem *> &phy_mem_list);

  ~MemPool();
};

}