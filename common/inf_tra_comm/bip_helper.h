#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/deque.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>


namespace colserve {

namespace bip = boost::interprocess;

template<typename T>
using bip_shm_allocator = 
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template<typename T>
using bip_deque = bip::deque<T, bip_shm_allocator<T> >;

using bip_mutex = bip::interprocess_mutex;
using bip_cond = bip::interprocess_condition;


}