#ifndef COLSERVE_SM_PARTITION_H
#define COLSERVE_SM_PARTITION_H


#include <common/util.h>

namespace colserve {

int GetGPUNumSM(int device);

void SetGlobalTPCMask(uint64_t mask_64);
void SetStreamTPCMask(CUstream s, uint64_t mask_64);

std::string CheckStreamSM(CUstream s);


}

#endif