#include <common/sm_partition.h>
#include "log_as_glog_sta.h"

namespace colserve {

#define CU_UUID_CONST static const
#define CU_CHAR(x) (char)((x) & 0xff)
// Define the symbol as exportable to other translation units, and
// initialize the value.  Inner set of parens is necessary because
// "bytes" array needs parens within the struct initializer, which
// also needs parens.  
#define CU_DEFINE_UUID(name, a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)          \
    CU_UUID_CONST CUuuid name =                                                \
    {                                                                          \
      {                                                                        \
        CU_CHAR(a), CU_CHAR((a) >> 8), CU_CHAR((a) >> 16), CU_CHAR((a) >> 24), \
        CU_CHAR(b), CU_CHAR((b) >> 8),                                         \
        CU_CHAR(c), CU_CHAR((c) >> 8),                                         \
        CU_CHAR(d0),                                                           \
        CU_CHAR(d1),                                                           \
        CU_CHAR(d2),                                                           \
        CU_CHAR(d3),                                                           \
        CU_CHAR(d4),                                                           \
        CU_CHAR(d5),                                                           \
        CU_CHAR(d6),                                                           \
        CU_CHAR(d7)                                                            \
      }                                                                        \
    }

CU_DEFINE_UUID(CU_ETID_SmDisableMask,
    0x8b7e90eb, 0x8cf2, 0x4a00, 0xb1, 0xd1, 0x08, 0xaa, 0x53, 0x55, 0x90, 0xdb);


__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

__global__ void get_sm_mask(int* buffer) {
  if (threadIdx.x == 0) {
    buffer[blockIdx.x] = get_smid();
  }
}

void SetStreamSM(CUstream s, unsigned int maskUpper, unsigned int maskLower) {
    const void* exportTable;
    CU_CALL(cuGetExportTable(&exportTable, &CU_ETID_SmDisableMask));
    CUresult (*set_mask)(CUstream, unsigned int, unsigned int) = \
        (CUresult (*)(CUstream, unsigned int, unsigned int))(*(unsigned long int*)(exportTable + (8*1)));
    CU_CALL(set_mask(s, maskUpper, maskLower));
    // printf("set mask: %08x, %08x\n", maskUpper, maskLower);
    LOG(INFO) << "stream " << s << " set mask " << std::setbase(16) << maskUpper << " " << maskLower;
}

std::string CheckStreamSM(CUstream s) {
  int* mask = nullptr;
  CUDA_CALL(cudaMallocHost(&mask, 1024 * sizeof(int)));

  get_sm_mask<<<1024, 1, 0, s>>>(mask);
  CUDA_CALL(cudaStreamSynchronize(s));

  std::set<int> used_sms;
  for (int i = 0; i < 1024; i++) {
    used_sms.insert(mask[i]);
  }
  std::stringstream ss;
  ss << "Stream " << s << ", use " << used_sms.size() << " SMs: ";

  for (auto sm : used_sms) {
    ss << sm << " ";
  }

  CUDA_CALL(cudaFreeHost(mask));

  return ss.str();
}

int GetGPUNumSM(int device) {
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  return deviceProp.multiProcessorCount;
}

}