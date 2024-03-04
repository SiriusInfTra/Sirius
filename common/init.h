#include "cuda_allocator.h"

namespace colserve {
namespace sta {

extern bool allocate_tensor_from_memory_pool;

// memory pool nbytes will be ignored if is not master
void InitMemoryPool(size_t memory_pool_nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);

}
}