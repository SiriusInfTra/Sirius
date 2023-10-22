#include "cuda_allocator.h"
#include "tensor_pool.h"


namespace colserve {
namespace sta {

// memory pool nbytes will be ignored if is not master
void Init(size_t memory_pool_nbytes = 0, bool master = false);

}
}