#ifndef COLSERVE_SM_PARTITION_H
#define COLSERVE_SM_PARTITION_H


#include <common/util.h>

namespace colserve {

void SetStreamSM(CUstream s, unsigned int maskUpper, unsigned int maskLower);

std::string CheckStreamSM(CUstream s);

int GetGPUNumSM(int device);

}

#endif