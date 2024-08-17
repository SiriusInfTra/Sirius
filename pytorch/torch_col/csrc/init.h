#ifndef TORCH_COL_INIT_H
#define TORCH_COL_INIT_H

#include <cstdint>

namespace torch_col {

void TorchColInit(int train_rank, int train_world_size);

void SMPartitionInit(uint64_t stream);

void TorchDistExtInit();

}

#endif