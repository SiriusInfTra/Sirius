#pragma once

#include <torch/csrc/distributed/c10d/reducer.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>

namespace torch_col {

class Reducer : public ::c10d::Reducer {
 public:
  template<typename ... T>
  Reducer(T&& ... args) : 
      ::c10d::Reducer(std::forward<T>(args)...) {}

  void finalize_dropped_batch();
 private:
};

class Logger : public ::c10d::Logger {
 public:
  template<typename ... T>
  Logger(T&& ... args) : 
      ::c10d::Logger(std::forward<T>(args)...) {}
};


}