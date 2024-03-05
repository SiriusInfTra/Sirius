#ifndef TORCH_COL_CONFIG_H
#define TORCH_COL_CONFIG_H

namespace torch_col {

extern int colocate_use_xsched;
  
extern int kill_batch_on_recv;

extern int has_colocated_infer_server;

extern int has_shared_tensor_server;

extern double shared_tensor_pool_gb;

void ConfigTorchCol();

}

#endif