#include <common/mempool.h> 
#include <common/log_as_glog_sta.h>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/lockfree/policies.hpp>
#include <boost/lockfree/queue.hpp>

#include <asm-generic/socket.h>
#include <sys/socket.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>

namespace colserve::sta {

std::unique_ptr<MemPool> MemPool::instance_ = nullptr;
std::atomic<bool> MemPool::is_init_ = false;

void HandleTransfer::SendHandles(int fd_list[], size_t len, bip::scoped_lock<bip::interprocess_mutex> &lock) {
  int socket_fd;
  struct sockaddr_un server_addr;

  // create socket and bind
  CHECK_GE(socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0), 0) << "[mempool] Socket creat fail.";

  bzero(&server_addr, sizeof(server_addr));
  server_addr.sun_family = AF_UNIX;

  unlink(master_name_.c_str());
  strncpy(server_addr.sun_path, master_name_.c_str(), master_name_.size());

  CHECK_EQ(bind(socket_fd, (struct sockaddr *)&server_addr, SUN_LEN(&server_addr)), 0) << "[mempool] Bind error.";

  // send to client
  ready_cond_->wait(lock);

  struct msghdr msg;
  struct iovec iov[1];


  std::vector<std::byte> control_un(CMSG_SPACE(len * sizeof(int)));

  struct cmsghdr *cmptr;
  struct sockaddr_un client_addr;
  bzero(&client_addr, sizeof(client_addr));
  client_addr.sun_family = AF_UNIX;
  strncpy(client_addr.sun_path, slave_name_.c_str(), slave_name_.size());

  msg.msg_control = control_un.data();
  msg.msg_controllen = control_un.size();

  cmptr = CMSG_FIRSTHDR(&msg);
  cmptr->cmsg_len = CMSG_LEN(len * sizeof(int));
  cmptr->cmsg_level = SOL_SOCKET;
  cmptr->cmsg_type = SCM_RIGHTS;


  memcpy(CMSG_DATA(cmptr), fd_list, len * sizeof(int));

  msg.msg_name = (void *)&client_addr;
  msg.msg_namelen = sizeof(struct sockaddr_un);
  iov[0].iov_base = (void *)"";
  iov[0].iov_len = len;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  ssize_t send_result = sendmsg(socket_fd, &msg, 0);
  CHECK_GE(send_result, 0) << "[mempool] Send msg fail.";

  // close socket
  unlink(master_name_.c_str());
  close(socket_fd);
}

void HandleTransfer::ReceiveHandle(int fd_list[], size_t len) {
  int socket_fd;
  struct sockaddr_un client_addr;
  CHECK_GE(socket_fd = socket(AF_UNIX, SOCK_DGRAM, 0), 0)  << "[mempool] Socket creat fail.";

  bzero(&client_addr, sizeof(client_addr));
  client_addr.sun_family = AF_UNIX;

  unlink(slave_name_.c_str());
  strncpy(client_addr.sun_path, slave_name_.c_str(), slave_name_.size());
  CHECK_EQ(bind(socket_fd, (struct sockaddr *)&client_addr, SUN_LEN(&client_addr)), 0) << "[mempool] Bind fail.";

  // recv from server
  {
    bip::scoped_lock lock{*ready_mutex_};
    ready_cond_->notify_all();
  }


  struct msghdr msg = {0};
  struct iovec iov[1];
  struct cmsghdr cm;

  std::vector<std::byte> control_un(CMSG_SPACE(len * sizeof(int)));

  struct cmsghdr *cmptr;
  ssize_t n;
  int recv_fd;
  char dummy_buf[1];

  msg.msg_control = control_un.data();
  msg.msg_controllen = control_un.size();

  iov[0].iov_base = (void *)dummy_buf;
  iov[0].iov_len = 1;

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  CHECK_GE(n = recvmsg(socket_fd, &msg, 0), 0)  << "[mempool] Recv msg fail.";

  CHECK((cmptr = CMSG_FIRSTHDR(&msg)) != nullptr) << "[mempool] Bad cmsg received.";
  CHECK_EQ(cmptr->cmsg_len, CMSG_LEN(sizeof(int) * len))  << "[mempool] Bad cmsg received.";

  memcpy(fd_list, CMSG_DATA(cmptr), sizeof(int) * len);
  CHECK_EQ(cmptr->cmsg_level, SOL_SOCKET) << "[mempool] Bad cmsg received.";
  CHECK_EQ(cmptr->cmsg_type, SCM_RIGHTS) << "[mempool] Bad cmsg received.";
  // close socket
  unlink(slave_name_.c_str());
  close(socket_fd);

}

void HandleTransfer::ExportWorker() {
  vmm_export_running_.store(true, std::memory_order_relaxed);
  LOG(INFO) << "[mempool] Master is now waitting for request vmm handles.";
  bip::scoped_lock request_lock{*request_mutex_};
  bip::scoped_lock ready_lock{*ready_mutex_};  
  while (true) {
    request_cond_->wait(request_lock);
 
    if (!vmm_export_running_.load(std::memory_order_relaxed)) {
      break;
    }
    LOG(INFO) << "[mempool] Master received request and began to send.";
    std::vector<int> fd_list(TRANSFER_CHUNK_SIZE);
    size_t chunk_base = 0;
    while (chunk_base < phy_mem_list_.size()) {
      size_t chunk_size =
          std::min(TRANSFER_CHUNK_SIZE, phy_mem_list_.size() - chunk_base);
      for (size_t k = 0; k < chunk_size; ++k) {
        CU_CALL(cuMemExportToShareableHandle(
            &fd_list[k], phy_mem_list_[chunk_base + k].cu_handle,
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      }
      LOG(INFO) << "[mempool] Master is sending handles: " << chunk_base << "/"
                << phy_mem_list_.size() << ".";
      SendHandles(fd_list.data(), chunk_size, ready_lock);
      for (size_t k = 0; k < chunk_size; ++k) {
        close(fd_list[k]);
      }
      chunk_base += chunk_size;
    }
  }
  LOG(INFO) << "[mempool] Master exit watting for request.";
}

HandleTransfer::HandleTransfer(bip::managed_shared_memory &shm,
                                              std::vector<PhyMem> &phy_mem_list,
                                              size_t mem_block_nbytes,
                                              size_t mem_block_num)
    : shared_memory_(shm), phy_mem_list_(phy_mem_list),
      mem_block_nbytes_(mem_block_nbytes) {
  char *gpu_id = getenv("CUDA_VISIBLE_DEVICES");
  CHECK(gpu_id != nullptr);
  char *username = getenv("USER");
  CHECK(username != nullptr);
  master_name_ = std::string("gpu-col-vmmipc-master-") + username + "-" + gpu_id;
  LOG(INFO) << "[mempool] Init VMM IPC, master_name = " << master_name_ << ".";
  slave_name_ = std::string("gpu-col-vmmipc-slave-") + username + "-" + gpu_id;
  LOG(INFO) << "[mempool] Init VMM IPC, slave_name = " << slave_name_ << ".";
  request_mutex_ = shared_memory_.find_or_construct<bip::interprocess_mutex>("HT_request_mutex_")();
  request_cond_ = shared_memory_.find_or_construct<bip::interprocess_condition>("HT_request_cond")();
  ready_mutex_ = shared_memory_.find_or_construct<bip::interprocess_mutex>("HT_ready_mutex_")();
  ready_cond_ = shared_memory_.find_or_construct<bip::interprocess_condition>("HT_ready_cond")();
  phy_mem_list.reserve(mem_block_num);
  Belong *belong_list = shared_memory_.find_or_construct<Belong>("HT_belong_list")[mem_block_num](Belong::kFree);
  phymem_queue::iterator *iter_list = shared_memory_.find_or_construct<phymem_queue::iterator>("HT_iter_list")[mem_block_num]();
  for (size_t k = 0; k < mem_block_num; ++k) {
    phy_mem_list.push_back(PhyMem{.index = k, .belong = &belong_list[k], .pos_queue = &iter_list[k]});
  }
}

void HandleTransfer::InitMaster() {
  CUmemAllocationProp prop = {
    .type = CU_MEM_ALLOCATION_TYPE_PINNED,
    .requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    .location = {
      .type = CU_MEM_LOCATION_TYPE_DEVICE,
      .id = 0
    }
  };
  auto start = std::chrono::steady_clock::now();
  for (auto &phy_mem : phy_mem_list_) {
    CU_CALL(cuMemCreate(&phy_mem.cu_handle, mem_block_nbytes_, &prop, 0));
  }
  auto end = std::chrono::steady_clock::now();
  LOG(INFO) << "[mempool] Alloc " 
            << phy_mem_list_.size() << " x " << ByteDisplay(mem_block_nbytes_) 
            << " block(s) costs " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms.";
  vmm_export_thread_.reset(new std::thread([&] { ExportWorker(); }));
}

void HandleTransfer::InitSlave() {
  std::vector<int> fd_list(TRANSFER_CHUNK_SIZE);
  size_t chunk_base = 0;
  {
    bip::scoped_lock ready_lock(*request_mutex_);
    request_cond_->notify_all();
  }
  while(chunk_base < phy_mem_list_.size()) {
    size_t chunk_size = std::min(TRANSFER_CHUNK_SIZE, phy_mem_list_.size() - chunk_base);
    LOG(INFO) << "[mempool] Slave is receving handles: " << chunk_base << "/" << phy_mem_list_.size() << ".";
    ReceiveHandle(fd_list.data(), chunk_size);
    for (size_t k = 0; k < chunk_size; ++k) {
      CU_CALL(cuMemImportFromShareableHandle(&phy_mem_list_[chunk_base + k].cu_handle, reinterpret_cast<void *>(fd_list[k]), 
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(fd_list[k]);
    }
    chunk_base += chunk_size;
  }
}

void HandleTransfer::ReleaseMaster() {
  vmm_export_running_.store(false, std::memory_order_relaxed);
  {
    boost::interprocess::scoped_lock ipc_lock(*request_mutex_);
    request_cond_->notify_all();
  }
  vmm_export_thread_->join();
}

MemPool::MemPool(size_t nbytes, bool cleanup): mempool_nbytes(nbytes) {
  CHECK_EQ(nbytes % MEM_BLOCK_NBYTES, 0);
  size_t mem_block_num = nbytes / MEM_BLOCK_NBYTES;
  auto *gpu_id = std::getenv("CUDA_VISIBLE_DEVICES");
  CHECK(gpu_id != nullptr);
  auto *username = std::getenv("USER");
  CHECK(username != nullptr);
  shared_memory_name_ = "gpu-colocate-shared-memory-" + std::string(username) +
                        "-" + std::string(gpu_id);

  CU_CALL(cuInit(0));
  if (cleanup) {
    bool ret = bip::shared_memory_object::remove(shared_memory_name_.c_str());
    LOG(INFO) << "[mempool] Remove shared_memory \"" << shared_memory_name_ << "\", ret = " << ret << ".";
  }
  shared_memory_ = bip::managed_shared_memory{bip::open_or_create,
                                              shared_memory_name_.c_str(),
                                              1 * 1024 * 1024 * 1024 /* 1G */};

  auto atomic_init = [&] {
    ref_count_ = shared_memory_.find_or_construct<int>("ME_ref_count")();
    mutex_ = shared_memory_.find_or_construct<bip::interprocess_mutex>("ME_mutex")();
    free_queue_ = shared_memory_.find_or_construct<phymem_queue>("ME_free_queue")(shared_memory_.get_segment_manager());
    allocated_nbytes_ = shared_memory_.find_or_construct<stats_arr>("ME_allocated_nbytes")();
    cached_nbytes_ = shared_memory_.find_or_construct<stats_arr>("ME_cached_nbytes_")();
  };
  shared_memory_.atomic_func(atomic_init);
  bip::scoped_lock locker(*mutex_);
  is_master_ = (*ref_count_)++ == 0;

  tranfer.reset(new HandleTransfer(shared_memory_, phy_mem_list_,
                                   MEM_BLOCK_NBYTES,
                                   mem_block_num));
  if (is_master_) {
    LOG(INFO) << "[mempool] Start to init master, nbytes = " << ByteDisplay(nbytes) << ".";
    tranfer->InitMaster();
    for(auto &&phy_mem : phy_mem_list_) {
      *phy_mem.pos_queue = free_queue_->insert(free_queue_->cend(), phy_mem.index);
    }
  } else {
    LOG(INFO) << "[mempool] Start to init slave.";
    tranfer->InitSlave();
  }
  is_init_ = true;
}

void MemPool::WaitSlaveExit() {
  if (is_master_) {
    auto getRefCount = [&] {
      bip::scoped_lock locker(*mutex_);
      return *ref_count_;
    };
    int ref_count;
    while ((ref_count = getRefCount()) > 1) {
      LOG(INFO) << "[mempool] master wait slave shutdown, ref_count = "
                << ref_count << ".";
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
}

MemPool::~MemPool() {
  is_init_ = false;
  if (is_master_) {
    WaitSlaveExit();
    tranfer->ReleaseMaster();
    bip::shared_memory_object::remove(shared_memory_name_.c_str());
    LOG(INFO) << "[mempool] free master.";
  } else {
    bip::scoped_lock locker(*mutex_);
    --(*ref_count_);
    LOG(INFO) << "[mempool] free slave.";
  }
}

void MemPool::AllocPhyMem(std::vector<PhyMem *> &phy_mem_list, Belong belong, size_t num_phy_mem) {
  CHECK_NE(belong, Belong::kFree);
  phy_mem_list.reserve(num_phy_mem);
  for (size_t k = 0; k < num_phy_mem && !free_queue_->empty(); ++k) {
    auto &phy_mem_ptr = phy_mem_list_[free_queue_->front()];
    *phy_mem_ptr.belong = belong;
    phy_mem_list.push_back(&phy_mem_ptr);
    free_queue_->pop_front();
  }
  cached_nbytes_->at(static_cast<size_t>(belong)).fetch_add(phy_mem_list.size() * MEM_BLOCK_NBYTES, std::memory_order_relaxed);
}

void MemPool::ClaimPhyMem(std::vector<PhyMem *> &phy_mem_list, Belong belong) {
  for (auto &phymem_ptr : phy_mem_list) {
    CHECK_EQ(*phymem_ptr->belong, Belong::kFree);
    free_queue_->erase(*phymem_ptr->pos_queue);
    *phymem_ptr->belong = belong;
  }
  cached_nbytes_->at(static_cast<size_t>(belong)).fetch_add(phy_mem_list.size() * MEM_BLOCK_NBYTES, std::memory_order_relaxed);
}

void MemPool::DeallocPhyMem(const std::vector<PhyMem *> &phy_mem_list) {
  if (phy_mem_list.empty()) { return; }
  Belong belong = *phy_mem_list.front()->belong;
  CHECK_NE(belong, Belong::kFree);
  for (auto &&phy_mem_ptr : phy_mem_list) {
    CHECK_EQ(*phy_mem_ptr->belong, belong);
    *phy_mem_ptr->belong = Belong::kFree;
    *phy_mem_ptr->pos_queue = free_queue_->insert(free_queue_->cend(), phy_mem_ptr->index);
  }
  cached_nbytes_->at(static_cast<size_t>(belong)).fetch_sub(phy_mem_list.size() * MEM_BLOCK_NBYTES, std::memory_order_relaxed);
}

MemPool &MemPool::Get() {
  if (instance_ == nullptr) {
    char *nbytes = getenv("COL_MEMPOOL_NBYTES");
    CHECK(nbytes != nullptr);
    char *cleanup = getenv("COL_MEMPOOL_CLEANUP");
    LOG(INFO) << "[mempool] Init with envs: COL_MEMPOOL_NBYTES=" << nbytes
              << ", COL_MEMPOOL_CLEANUP=" << cleanup << ".";
    instance_.reset(
        new MemPool(std::stoul(nbytes),
                    cleanup == nullptr ? false : std::stoi(cleanup) == 1));
  }
  return *instance_;
}

bool MemPool::IsInit() {
  return is_init_.load(std::memory_order_relaxed);
}

void MemPool::PrintStatus() {
  for (Belong belong : {Belong::kInfer, Belong::kTrain}) {
    LOG(INFO) << belong << " Allocate: " << ByteDisplay(GetAllocatedNbytes(belong));
    LOG(INFO) << belong << " Cached: " << ByteDisplay(GetCachedNbytes(belong));
  }
  LOG(INFO) << "[mempool] nbytes = " << ByteDisplay(mempool_nbytes);
  LOG(INFO) << "[mempool] total phy block = " << phy_mem_list_.size();
  LOG(INFO) << "[mempool] free phy block = " << std::count_if(phy_mem_list_.cbegin(), phy_mem_list_.cend(), [](auto &phy_mem) { return *phy_mem.belong == Belong::kFree; });
  LOG(INFO) << "[mempool] train phy block = " << std::count_if(phy_mem_list_.cbegin(), phy_mem_list_.cend(), [](auto &phy_mem) { return *phy_mem.belong == Belong::kTrain; });
  LOG(INFO) << "[mempool] infer phy block = " << std::count_if(phy_mem_list_.cbegin(), phy_mem_list_.cend(), [](auto &phy_mem) { return *phy_mem.belong == Belong::kInfer; });
  LOG(INFO) << "[mempool] freelist len = " << free_queue_->size();
}

}  // namespace colserve::sta