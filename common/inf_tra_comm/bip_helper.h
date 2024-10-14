#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/deque.hpp>
#include <boost/interprocess/containers/set.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>


namespace colserve {

namespace bip = boost::interprocess;

template<typename T>
using bip_shm_allocator = 
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template<typename T>
using bip_deque = bip::deque<T, bip_shm_allocator<T> >;

template<typename T>
using bip_set = bip::set<T, std::less<T>, bip_shm_allocator<T> >;

template<typename K, typename V>
using bip_map = bip::map<K, V, std::less<K>, 
    bip_shm_allocator<std::pair<const K, V> > >;

using bip_mutex = bip::interprocess_mutex;
using bip_cond = bip::interprocess_condition;
using bip_sem = bip::interprocess_semaphore;
using bip_named_sem = bip::named_semaphore;


template <typename ContainerType>
struct is_bip_container : std::false_type {};

template <typename ElemType>
struct is_bip_container<bip_deque<ElemType>> : std::true_type {};

template <typename ElemType>
struct is_bip_container<bip_set<ElemType>> : std::true_type {};

template <typename KeyType, typename ValueType>
struct is_bip_container<bip_map<KeyType, ValueType>> : std::true_type {};


template <typename SyncVarType>
struct is_bip_sync_var : std::false_type {};

template <>
struct is_bip_sync_var<bip_mutex> : std::true_type {};

template<>
struct is_bip_sync_var<bip_cond> : std::true_type {};

template<>
struct is_bip_sync_var<bip_sem> : std::true_type {};


template<typename  T>
using cont_allocator_t = typename T::allocator_type;



} // namespace colserve