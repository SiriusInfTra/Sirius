#ifndef TORCH_COL_LOGGING_H
#define TORCH_COL_LOGGING_H

#include <iostream>
#include <memory>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <Python.h>

namespace torch_col {

#define TORCH_COL_PREDICT_TRUE(x) __builtin_expect(!!(x), 1)
#define TORCH_COL_PREDICT_FALSE(x) __builtin_expect(x, 0)

#define TORCH_COL_LEVEL_INFO 1
#define TORCH_COL_LEVEL_WARNING 2
#define TORCH_COL_LEVEL_FATAL 3
#define TORCH_COL_LOG_LEVEL(LEVEL) TORCH_COL_LEVEL_ ## LEVEL


class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno, int level) : level_(level) {
    std::time_t t = std::time(nullptr);
    stream_ << "[" << level_strings_[level] << " " << std::put_time(std::localtime(&t), "%H:%M:%S") << " " 
            << file << ":" << lineno<< "] ";
  }
  ~LogMessage() {
    if (level_ == TORCH_COL_LEVEL_INFO) {
      std::cout << stream_.str() << std::endl; 
    } else {
      std::cerr << stream_.str() << std::endl; 
    }
  }
  std::ostringstream& stream() { return stream_; }

 protected:
  static const char* level_strings_[];
  int level_;
  std::ostringstream stream_;
};

class LogFatal : public LogMessage {
 public:
  LogFatal(const char* file, int lineno) : LogMessage(file, lineno, TORCH_COL_LOG_LEVEL(FATAL)) {}
  [[noreturn]] ~LogFatal() { 
    std::cerr << stream_.str() << std::endl; 
    abort();
  }
};

#define LOG(INFO) ::torch_col::LogMessage(__FILE__, __LINE__, TORCH_COL_LOG_LEVEL(INFO)).stream()


template <typename X, typename Y>
std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs. " << y << ") ";  // CHECK_XX(x, y) requires x and y can be serialized to
                                            // string. Use CHECK(x OP y) otherwise.
  return std::make_unique<std::string>(os.str());
}

#define TORCH_COL_CHECK_OP_FUNC(name, op) \
  template <typename X, typename Y> \
  inline __attribute__((always_inline)) \
  std::unique_ptr<std::string> LogCheck##name(const X&x, const Y&y) { \
    if (TORCH_COL_PREDICT_TRUE(x op y)) return nullptr; \
    return LogCheckFormat(x, y); \
  } \
  inline __attribute__((always_inline)) \
  std::unique_ptr<std::string> LogCheck##name(int x, int y) { \
    return LogCheck##name<int, int>(x, y); \
  }

TORCH_COL_CHECK_OP_FUNC(_LT, <)
TORCH_COL_CHECK_OP_FUNC(_GT, >)
TORCH_COL_CHECK_OP_FUNC(_LE, <=)
TORCH_COL_CHECK_OP_FUNC(_GE, >=)
TORCH_COL_CHECK_OP_FUNC(_EQ, ==)
TORCH_COL_CHECK_OP_FUNC(_NE, !=)

#define CHECK(x)                                      \
  if (TORCH_COL_PREDICT_FALSE(!(x)))                  \
  ::torch_col::LogFatal(__FILE__, __LINE__).stream()  \
      << "Check failed: (" #x << ") is false: "


#define TORCH_COL_CHECK_BINARY_OP(name, op, x, y)                          \
  if (auto __log__err =  ::torch_col::LogCheck##name(x, y))                \
   ::torch_col::LogFatal(__FILE__, __LINE__).stream()                      \
      << "Check failed: " << #x " " #op " " #y << *__log__err << ": "

#define CHECK_LT(x, y) TORCH_COL_CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) TORCH_COL_CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) TORCH_COL_CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) TORCH_COL_CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) TORCH_COL_CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) TORCH_COL_CHECK_BINARY_OP(_NE, !=, x, y)

}

#endif