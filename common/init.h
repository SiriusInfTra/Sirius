#include "cuda_allocator.h"

namespace colserve {
namespace sta {

// memory pool nbytes will be ignored if is not master
void Init(size_t memory_pool_nbytes, bool cleanup, bool observe, FreeListPolicyType free_list_policy);

}
}