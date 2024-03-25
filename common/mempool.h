#pragma once

#include <common/util.h>

#include <boost/interprocess/containers/list.hpp>
#include <boost/lockfree/policies.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/circular_buffer.hpp>

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iterator>
#include <ostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <thread>

#include <atomic>


namespace colserve::sta {
const static constexpr size_t MEM_BLOCK_NBYTES = 32_MB;

namespace detail {
constexpr size_t alignment = 1024;
inline size_t GetAlignedNbytes(size_t nbytes) {
  static_assert((alignment & (alignment - 1)) == 0, "alignment must be power of 2");
  return (nbytes + (alignment - 1)) & (~(alignment - 1));
}

template<size_t align>
inline size_t AlignedNBytes(size_t nbytes) {
  static_assert((align & (align - 1)) == 0, "alignment must be power of 2");
  return (nbytes + (align - 1)) & (~(align - 1));
}
}


enum class Belong {
  kTrain, 
  kInfer, 
  kUsedNum, 
  kFree
};

inline std::string ToString(Belong belong) {
  switch (belong) {
  case Belong::kInfer:
    return "kInfer";
  case Belong::kTrain:
    return "kTrain";
  case Belong::kFree:
    return "kFree";
  default:
    return "Unknown(" + std::to_string(static_cast<size_t>(belong)) + ")";
  }
}

inline std::ostream & operator<<(std::ostream &os, const Belong &belong)  {
  switch (belong) {
  case Belong::kInfer:
    os << "kInfer";
    break;
  case Belong::kTrain:
    os << "kTrain";
    break;
  case Belong::kFree:
    os << "kFree";
    break;
  default:
    os << "Unknown(" << static_cast<size_t>(belong) << ")";
    break;
  }
  return os;
}

namespace bip = boost::interprocess;
using phymem_queue = bip::list<size_t, bip::allocator<size_t, bip::managed_shared_memory::segment_manager>>;


struct PhyMem {
  const size_t index;
  CUmemGenericAllocationHandle cu_handle;
  Belong * const belong;
  phymem_queue::iterator * const pos_queue;
};

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
  HandleTransfer(bip::managed_shared_memory &shm, 
                 std::vector<PhyMem> &phy_mem_list, 
                 size_t mem_block_nbytes, 
                 size_t mem_block_num);

  void InitMaster();
  void InitSlave();
  void ReleaseMaster();
};


class MemPool {
  using ring_buffer = 
      boost::circular_buffer<size_t, bip::allocator<size_t, bip::managed_shared_memory::segment_manager>>;
  using stats_arr = std::array<std::atomic<size_t>, static_cast<size_t>(Belong::kUsedNum)>;
  using phymem_callback = std::function<void(const std::vector<PhyMem*> &phymem_arr)>;
  template<typename T> friend class shm_handle;
private:
  static std::unique_ptr<MemPool> instance_;
  static std::atomic<bool> is_init_;

  bip::managed_shared_memory  shared_memory_;
  bip::interprocess_mutex     *mutex_;
  int                         *ref_count_;
  std::string                 shared_memory_name_;

  std::unique_ptr<HandleTransfer> tranfer;
  std::vector<PhyMem> phy_mem_list_;
  phymem_queue *free_queue_;
  stats_arr *allocated_nbytes_;
  stats_arr *mem_page_nbytes;
  
  bool is_master_;
public:
  static MemPool &Get();
  static bool IsInit();

  const size_t mempool_nbytes;


  MemPool(size_t nbytes, bool cleanup);

  void WaitSlaveExit();

  inline std::vector<PhyMem> &GetPhyMemList() { return phy_mem_list_; }

  inline bip::managed_shared_memory &GetSharedMemory() { return shared_memory_; }

  inline bip::interprocess_mutex &GetMutex() { return *mutex_; }

  void AllocPhyMem(std::vector<PhyMem *> &phy_mem_list, Belong belong, size_t num_phy_mem);

  void ClaimPhyMem(std::vector<PhyMem *> &phy_mem_list, Belong belong);

  void DeallocPhyMem(const std::vector<PhyMem *> &phy_mem_list);

  void DumpMemPool(std::ostream &out) {
    out << "index,belong,cu_handle\n";
    for(auto &phy_mem : phy_mem_list_) {
      out << phy_mem.index << "," << *phy_mem.belong << "," << phy_mem.cu_handle << "\n";
    }
    out << std::flush;
  }

  inline size_t GetAllocatedNbytes(Belong belong) {
    return allocated_nbytes_->at(static_cast<size_t>(belong)).load(std::memory_order_relaxed);
  }

  inline size_t AddAllocatedNbytes(long nbytes, Belong belong) {
    return allocated_nbytes_->at(static_cast<size_t>(belong)).fetch_add(nbytes, std::memory_order_relaxed);
  }

  inline size_t SubAllocatedNbytes(long nbytes, Belong belong) {
    return allocated_nbytes_->at(static_cast<size_t>(belong)).fetch_sub(nbytes, std::memory_order_relaxed);
  }

  inline size_t GetMemPageNBytes(Belong belong) {
    return mem_page_nbytes->at(static_cast<size_t>(belong)).load(std::memory_order_relaxed);
  }

  void PrintStatus();

  ~MemPool();
};

}