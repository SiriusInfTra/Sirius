#ifndef COLSERVE_SM_PARTITION_H
#define COLSERVE_SM_PARTITION_H


#include <common/util.h>

namespace colserve {

int GetGPUNumSM(int device);

void SetStreamSMMask(CUstream s, uint64_t mask_64);

std::string CheckStreamSM(CUstream s);


}

#endif