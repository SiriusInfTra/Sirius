#ifndef TORCH_COL_INIT_H
#define TORCH_COL_INIT_H

namespace torch_col {

void TorchColInit(int train_rank, int train_world_size);

void InitSMPartition();

}

#endif