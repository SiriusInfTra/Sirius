#ifndef COLSERVE_MEMORY_QUEUE_H
#define COLSERVE_MEMORY_QUEUE_H

#include <iostream>
#include <array>
#include <mutex>
#include <pthread.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <queue>


namespace colserve {

namespace {
template<typename T, typename = std::void_t<>>
struct MQElement : public std::false_type {
};

template<typename NumType>
struct MQElement<NumType> : public std::true_type {
  static_assert(std::is_arithmetic<NumType>::value, "invalid type");
  static void to_memory(void* memory, const NumType &data) {
    reinterpret_cast<NumType*>(memory)[0] = data;
  }
  static void to_data(void* memory, NumType &data) {
    data = reinterpret_cast<NumType*>(memory)[0];
  }
  static constexpr size_t size() {
    return sizeof(NumType);
  }
};

template<typename T, size_t N>
struct MQElement<std::array<T, N>> : public std::true_type {
  using Array = std::array<T, N>;
  static_assert(std::is_arithmetic<T>::value || std::is_same<char, T>::value, "invalid type");
  static void to_memory(void* memory, const Array &data) {
    memcpy(memory, data.data(), size());
  }
  static void to_data(void* memory, Array &data) {
    memcpy(data.data(), memory, size());
  }
  static constexpr size_t size() {
    return sizeof(T) * N;
  }
};
}

template <typename T>
class MemoryQueue {
 public :
  static_assert(MQElement<T>::value, "invalid type");
  static constexpr auto to_memory = MQElement<T>::to_memory;
  static constexpr auto to_data = MQElement<T>::to_data;
  static constexpr auto elem_size = MQElement<T>::size;
  MemoryQueue(const std::string &name, bool is_server) 
      : name_(name), is_server_(is_server) {
    // std::string shm_name = "colserve-mq-" + name;
    auto shm_name = GetShmName(-1);
    if (is_server) {
      int shm_meta = shm_open(shm_name.c_str(), O_RDWR | O_CREAT, 0666);
      CHECK_NE(shm_meta, -1);
      CHECK_NE(ftruncate(shm_meta, sizeof(MetaData)), -1);
      meta_data_ = static_cast<MetaData*>(mmap(nullptr, sizeof(MetaData), 
          PROT_READ | PROT_WRITE, MAP_SHARED, shm_meta, 0));
      CHECK_NE(meta_data_, MAP_FAILED);

      pthread_mutexattr_t attr;
      CHECK_EQ(pthread_mutexattr_init(&attr), 0);
      CHECK_EQ(pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED), 0);
      CHECK_EQ(pthread_mutex_init(&meta_data_->mutex_, &attr), 0);
      // CHECK_EQ(pthread_mutex_lock(&meta_data_->mutex_), 0);
      CHECK_EQ(sem_init(&meta_data_->len_sem_, 1, 0), 0);
      meta_data_->version_ = 0;
      meta_data_->cap_ = kCap;
      meta_data_->size_ = sizeof(MemoryData) + kCap * elem_size();
      // meta_data_->shm_data_name_ = "colserve-mq-" + name + "-" + std::to_string(meta_data_->version_);
      // meta_data_->shm_data_name_ = GetShmName(meta_data_->version_);
      memcpy(meta_data_->shm_data_name_, GetShmName(meta_data_->version_).c_str(), 
          GetShmName(meta_data_->version_).size() + 1);

      int shm_data = shm_open(meta_data_->shm_data_name_, O_RDWR | O_CREAT, 0666);
      CHECK_NE(shm_data, -1);
      CHECK_NE(ftruncate(shm_data, meta_data_->size_), -1);
      memory_data_ = static_cast<MemoryData*>(mmap(nullptr, meta_data_->size_, 
          PROT_READ | PROT_WRITE, MAP_SHARED, shm_data, 0));
      CHECK_NE(memory_data_, MAP_FAILED);
      memory_data_->version_ = 0;
      memory_data_->cap_ = meta_data_->cap_;
      memory_data_->size_ = meta_data_->size_;
      memory_data_->idx_ = 0;
      memory_data_->len_ = 0;
      // CHECK_EQ(pthread_mutex_unlock(&meta_data_->mutex_), 0);
    } else {
      int shm_meta = shm_open(shm_name.c_str(), O_RDWR, 0666);
      CHECK(shm_meta != -1);
      meta_data_ = static_cast<MetaData*>(mmap(nullptr, sizeof(MetaData), 
          PROT_READ | PROT_WRITE, MAP_SHARED, shm_meta, 0));
      CHECK_NE(meta_data_, MAP_FAILED);

      Lock();
      int shm_data = shm_open(meta_data_->shm_data_name_, O_RDWR, 0666);
      CHECK(shm_data != -1);
      memory_data_ = static_cast<MemoryData*>(mmap(nullptr, meta_data_->size_, 
          PROT_READ | PROT_WRITE, MAP_SHARED, shm_data, 0));
      CHECK_NE(memory_data_, MAP_FAILED);
      CHECK(CheckMetaDataUnlocked());
      Unlock();
    }
  }
  void Put(const T &data) {
    Lock();
    UpdateMemoryDataUnlocked();
    if (memory_data_->len_ == meta_data_->cap_) {
      // TODO: double mmap to speed put when mq is full, async extend shm 
      LOG(INFO) << "[MermoryQueue] " << name_ << " is full,"
                << " version=" << meta_data_->version_
                << " cap=" << meta_data_->cap_;
      meta_data_->version_++;
      meta_data_->cap_ = meta_data_->cap_ * 2;
      meta_data_->size_ = sizeof(MemoryData) + meta_data_->cap_ * elem_size();
      // meta_data_->shm_data_name_ = GetShmName(meta_data_->version_);
      memcpy(meta_data_->shm_data_name_, GetShmName(meta_data_->version_).c_str(), 
          GetShmName(meta_data_->version_).size() + 1);

      int shm_data = shm_open(meta_data_->shm_data_name_, O_RDWR | O_CREAT, 0666);
      CHECK_NE(shm_data, -1);
      CHECK_NE(ftruncate(shm_data, meta_data_->size_), -1);
      auto new_memory_data = static_cast<MemoryData*>(mmap(nullptr, meta_data_->size_, 
          PROT_READ | PROT_WRITE, MAP_SHARED, shm_data, 0));
      CHECK_NE(new_memory_data, MAP_FAILED);
      new_memory_data->version_ = meta_data_->version_;
      new_memory_data->idx_ = 0;
      new_memory_data->len_ = memory_data_->len_;
      for (size_t i = 0; i < memory_data_->len_; i++) {
        memcpy(&new_memory_data->data_[i], 
            &memory_data_->data_[(memory_data_->idx_ + i) % memory_data_->cap_], elem_size());
      }
      CHECK_EQ(munmap(memory_data_, memory_data_->size_), 0);
      shm_unlink(GetShmName(memory_data_->version_).c_str());
      memory_data_ = new_memory_data;
    }
    // memory_data_->data_[(memory_data_->idx_ + memory_data_->len_) % memory_data_->cap_] = data;
    to_memory(&memory_data_->data_[(memory_data_->idx_ + memory_data_->len_) % memory_data_->cap_], data);
    memory_data_->len_++;
    CHECK_EQ(sem_post(&meta_data_->len_sem_), 0);
    Unlock();
  }
  T BlockGet() {
    CHECK_EQ(sem_wait(&meta_data_->len_sem_), 0);
    T data{};
    GetSemUnwaited(data);
    return data;
  }
  bool TimedGet(T &data, size_t timeout_ms) {
    timespec ts;
    CHECK_NE(clock_gettime(CLOCK_REALTIME, &ts), -1);
    size_t sec = timeout_ms / 1000;
    size_t ns = timeout_ms % 1000 * 1000000;
    ts.tv_sec += sec;
    ts.tv_nsec += ns;
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000;
    }
    auto err = sem_timedwait(&meta_data_->len_sem_, &ts);
    CHECK(err == 0 || errno == ETIMEDOUT);
    if (err != 0) {
      return false;
    }
    GetSemUnwaited(data);
    return true;
  }
  bool TryGet(T &data) {
    auto err = sem_trywait(&meta_data_->len_sem_);
    CHECK(err == 0 || errno == EAGAIN);
    if (err != 0) {
      return false;
    } 
    GetSemUnwaited(data);
    return true;
  }
  void Clear() {
    Lock();
    memory_data_->idx_ = 0;
    memory_data_->len_ = 0;
    CHECK_EQ(sem_destroy(&meta_data_->len_sem_), 0);
    CHECK_EQ(sem_init(&meta_data_->len_sem_, 1, 0), 0);
    Unlock();
  }
  
  ~MemoryQueue() {
    auto meta_data_shm_name = GetShmName(-1);
    auto memory_data_shm_name = std::string(meta_data_->shm_data_name_);
    // DLOG(INFO) << "delete mq " << name_ << " " << meta_data_->shm_data_name_;
    CHECK_EQ(munmap(memory_data_, meta_data_->size_), 0);
    CHECK_EQ(munmap(meta_data_, sizeof(MetaData)), 0);
    if (is_server_) {
      shm_unlink(memory_data_shm_name.c_str());
      shm_unlink(meta_data_shm_name.c_str());
    }
    // DLOG(INFO) << "delete mq succ";
  }

 private :
  struct MetaData
  {
    size_t version_;
    size_t size_;
    size_t cap_;
    sem_t len_sem_;
    pthread_mutex_t mutex_;
    char shm_data_name_[128];
  };

  struct MemoryData {
    size_t version_;
    size_t size_;
    size_t cap_;

    size_t len_;
    size_t idx_;
    T data_[0];
  };

  static constexpr size_t kCap = 128;


  std::string GetShmName(size_t version) {
    if (version == static_cast<size_t>(-1)) {
      return "colserve-mq-" + std::string(getuid()) + "-" + name_;
    } else {
      return "colserve-mq-" + std::string(getuid()) + "-" + name_ + "-" + std::to_string(version);
    }
  }
  void Lock() {
    CHECK_EQ(pthread_mutex_lock(&meta_data_->mutex_), 0);
  }
  void Unlock() {
    CHECK_EQ(pthread_mutex_unlock(&meta_data_->mutex_), 0);
  }
  void GetSemUnwaited(T &data) {
    Lock();
    UpdateMemoryDataUnlocked();
    CHECK_GT(memory_data_->len_, 0);
    // T data = memory_data_->data_[memory_data_->idx_];
    to_data(&memory_data_->data_[memory_data_->idx_], data);
    memory_data_->idx_ = (memory_data_->idx_ + 1) % memory_data_->cap_;
    memory_data_->len_--;
    Unlock();
  }
  void UpdateMemoryDataUnlocked() {
    if (memory_data_->version_ == meta_data_->version_) {
      return;
    }
    int shm_data = shm_open(meta_data_->shm_data_name_, O_RDWR, 0666);
    CHECK(shm_data != -1);
    auto new_memory_data = static_cast<MemoryData*>(mmap(nullptr, meta_data_->size_, 
        PROT_READ | PROT_WRITE, MAP_SHARED, shm_data, 0));
    CHECK_NE(new_memory_data, MAP_FAILED);
    CHECK_EQ(munmap(memory_data_, memory_data_->size_), 0);
    shm_unlink(GetShmName(memory_data_->version_).c_str());
    memory_data_ = new_memory_data;
    CHECK(CheckMetaDataUnlocked());
  }
  bool CheckMetaDataUnlocked() {
    if (memory_data_->version_ == meta_data_->version_
        && memory_data_->size_ == meta_data_->size_
        && memory_data_->cap_ == meta_data_->cap_) {
      return true;
    } else {
      return false;
    }
  }

  bool is_server_;
  std::string name_;
  MetaData* meta_data_;
  MemoryData* memory_data_;
};


template <typename T>
class BlockQueue {
 public:
  BlockQueue() {
    CHECK_EQ(sem_init(&sem_, 0, 0), 0);
  }
  void Put(const T &data) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(data);
    CHECK_EQ(sem_post(&sem_), 0);
  };
  T BlockGet(){
    CHECK_EQ(sem_wait(&sem_), 0);
    std::lock_guard<std::mutex> lock(mutex_);
    T data = queue_.front();
    queue_.pop();
    return data;
  };
  bool TimedGet(T &data, size_t timeout_ms){
    timespec ts;
    CHECK_NE(clock_gettime(CLOCK_REALTIME, &ts), -1);
    size_t sec = timeout_ms / 1000;
    size_t ns = timeout_ms % 1000 * 1000000;
    ts.tv_sec += sec;
    ts.tv_nsec += ns;
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000;
    }
    auto err = sem_timedwait(&sem_, &ts);
    CHECK(err == 0 || errno == ETIMEDOUT);
    if (err != 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    data = queue_.front();
    queue_.pop();
    return true;
  };
  bool TryGet(T &data){
    int err = sem_trywait(&sem_);
    CHECK(err == 0 || errno == EAGAIN);
    if (err != 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    data = queue_.front();
    queue_.pop();
    return true;
  };
 private:
  sem_t sem_;
  std::mutex mutex_;
  std::queue<T> queue_;
};

} // namespace colserve

#endif