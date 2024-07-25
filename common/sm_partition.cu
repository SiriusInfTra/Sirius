#include <common/sm_partition.h>
#include "log_as_glog_sta.h"

#include <error.h>
#include <errno.h>
#include <cstdint>
#include <unistd.h>

namespace colserve {

namespace {
// ref: http://rtsrv-eth.telenet.unc.edu/cgit/cgit.cgi/libsmctrl.git/

using uint128_t = __uint128_t;

#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

// Tested working on CUDA x86_64 11.0-12.2.
// Tested not working on aarch64 or x86_64 10.2
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3
static uint64_t g_sm_mask = 0;
static __thread uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;
static void launchCallback(void *ukwn, int domain, int cbid, const void *in_params) {
	// The third 8-byte element in `in_parms` is a pointer to the stream struct.
	// This exists even when in_params < 0x50. This could be used to implement
	// stream masking without the manual offsets specified elsewhere (store a
	// table of stream pointers to masks and do a lookup here).
	// It could also be used (although not as easily) to support global and next
	// masking on old CUDA versions, but that would require hooking earlier in the
	// launch process (before the stream mask is applied).
	if (*(uint32_t*)in_params < 0x50)
		abort(1, 0, "Unsupported CUDA version for callback-based SM masking. Aborting...");
	// The eighth 8-byte element in `in_params` is a pointer to a struct which
	// contains a pointer to the TMD as its first element. Note that this eighth
	// pointer must exist---it only exists when the first 8-byte element of
	// `in_params` is at least 0x50 (checked above).
	void* tmd = **((uintptr_t***)in_params + 8);
	if (!tmd)
		abort(1, 0, "TMD allocation appears NULL; likely forward-compatibilty issue.\n");

	//fprintf(stderr, "cta: %lx\n", *(uint64_t*)(tmd + 74));
	// TODO: Check for supported QMD version (>XXX, <4.00)
	// TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 16 bytes (rather than 8 bytes) wide. It also requires an enable bit at +31bits.
	uint32_t *lower_ptr = (uint32_t*)(tmd + 84);
	uint32_t *upper_ptr = (uint32_t*)(tmd + 88);

	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;	
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}
	//fprintf(stderr, "lower mask: %x\n", *lower_ptr);
	//fprintf(stderr, "upper mask: %x\n", *upper_ptr);
}


static void setup_sm_control_11() {
	int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
	int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
	uintptr_t* tbl_base;
	uint32_t my_hndl;
	// Avoid race conditions (setup can only be called once)
	if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
		return;

	cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
	uintptr_t subscribe_func_addr = *(tbl_base + 3);
	uintptr_t enable_func_addr = *(tbl_base + 6);
	subscribe = (typeof(subscribe))subscribe_func_addr;
	enable = (typeof(enable))enable_func_addr;
	int res = 0;
	res = subscribe(&my_hndl, launchCallback, NULL);
	if (res)
		abort(1, 0, "Error subscribing to launch callback. CUDA returned error code %d.", res);
	res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
	if (res)
		abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
}


#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
#define CU_9_0_MASK_OFF_TX2 0x128 // CUDA 9.0 is slightly different on the TX2
// CUDA 9.0 and 9.1 use the same offset
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x24c
// CUDA 10.0, 10.1 and 10.2 use the same offset
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
#define CU_12_0_MASK_OFF 0x4cc
#define CU_12_3_MASK_OFF 0x49c
// CUDA 12.0 and 12.1 use the same offset
// CUDA 12.3 try to use 0x4cc - 48

// Set default mask for all launches
void libsmctrl_set_global_mask(uint64_t mask) {
	int ver;
	cuDriverGetVersion(&ver);
	if (ver == 10020) {
		// if (!g_sm_control)
		// 	setup_g_sm_control_10();
		// g_sm_control->mask = mask;
		// g_sm_control->enabled = 1;
		abort(1, ENOSYS, "we detected CUDA 10.2, although it can be supported originally");
	} else if (ver > 10020) {
		if (!sm_control_setup_called)
			setup_sm_control_11();
		g_sm_mask = mask;
	} else { // < CUDA 10.2
		abort(1, ENOSYS, "Global masking requires at least CUDA 10.2; "
		                 "this application is using CUDA %d.%d",
		                 ver / 1000, (ver % 100));
	}
}

// Set mask for next launch from this thread
void libsmctrl_set_next_mask(uint64_t mask) {
	if (!sm_control_setup_called)
		setup_sm_control_11();
	g_next_sm_mask = mask;
}

/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

// Offsets for the stream struct on x86_64
#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
// CUDA 9.0 and 9.1 use the same offset
// 9.1 tested on 390.157
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x244
// CUDA 10.0, 10.1 and 10.2 use the same offset
// 10.1 tested on 418.113
// 10.2 tested on 440.100, 440.82, 440.64, and 440.36
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
// 11.4 tested on 470.223.02
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
// 11.8 tested on 520.56.06
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset
// 12.0 tested on 525.147.05
#define CU_12_2_MASK_OFF 0x4e4
// 12.2 tested on 535.129.03
#define CU_12_3_MASK_OFF 0x49c
// 12.3 tested on 545.23.08, 0x49c=0x49c-72


// Offsets for the stream struct on aarch64
// All tested on Nov 13th, 2023
#define CU_9_0_MASK_OFF_JETSON 0x128 // Tested on TX2
#define CU_10_2_MASK_OFF_JETSON 0x24c // Tested on TX2 and Jetson Xavier
#define CU_11_4_MASK_OFF_JETSON 0x394 // Tested on Jetson Orin

// Used up through CUDA 11.8 in the stream struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
};

// Used starting with CUDA 12.0 in the stream struct
struct stream_sm_mask_v2 {
	uint32_t enabled;
	uint32_t mask[4];
};

// Should work for CUDA 8.0 through 12.2
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask);
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	uint128_t full_mask = -1;
	full_mask <<= 64;
	full_mask |= mask;
	libsmctrl_set_stream_mask_ext(stream, full_mask);
}

void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask = NULL;
	struct stream_sm_mask_v2* hw_mask_v2 = NULL;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
#if __x86_64__
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010: {
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
		break;
	}
	case 9020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_2_MASK_OFF);
		break;
	case 10000:
	case 10010:
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_0_MASK_OFF);
		break;
	case 11000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_0_MASK_OFF);
		break;
	case 11010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_1_MASK_OFF);
		break;
	case 11020:
	case 11030:
	case 11040:
	case 11050:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_2_MASK_OFF);
		break;
	case 11060:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_6_MASK_OFF);
		break;
	case 11070:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_7_MASK_OFF);
		break;
	case 11080:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF);
		break;
	case 12000:
	case 12010:
		hw_mask_v2 = (struct stream_sm_mask_v2*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	case 12020:
		hw_mask_v2 = (struct stream_sm_mask_v2*)(stream_struct_base + CU_12_2_MASK_OFF);
		break;
	case 12030:
		hw_mask_v2 = (struct stream_sm_mask_v2*)(stream_struct_base + CU_12_3_MASK_OFF);
		break;
#elif __aarch64__
	case 9000: {
		// Jetson TX2 offset is slightly different on CUDA 9.0.
		// Only compile the check into ARM64 builds.
		// TODO: Always verify Jetson-board-only on aarch64.
		int is_parker;
		const char* err_str;
		if ((is_parker = detect_parker_soc()) < 0) {
			cuGetErrorName(-is_parker, &err_str);
			abort(1, 0, "While performing platform-specific "
			            "compatibility checks for stream masking, "
			            "CUDA call failed with error '%s'.", err_str);
		}

		if (!is_parker)
			abort(1, 0, "Not supported on non-Jetson aarch64.");
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF_JETSON);
		break;
	}
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_2_MASK_OFF_JETSON);
		break;
	case 11040:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_4_MASK_OFF_JETSON);
		break;
#endif
	}

	// For experimenting to determine the right mask offset, set the MASK_OFF
	// environment variable (positive and negative numbers are supported)
	char* mask_off_str = getenv("MASK_OFF");
	if (mask_off_str) {
		int off = atoi(mask_off_str);
		fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.2 base %#x "
				"(total off: %#x)\n", off, CU_12_2_MASK_OFF, CU_12_2_MASK_OFF + off);
		if (CU_12_2_MASK_OFF + off < 0)
			abort(1, 0, "Total offset cannot be less than 0! Aborting...");
		// +4 bytes to convert a mask found with this for use with hw_mask
		hw_mask_v2 = (struct stream_sm_mask_v2*)(stream_struct_base + CU_12_2_MASK_OFF + off);
	}

	// Mask layout changed with CUDA 12.0 to support large Hopper/Ada GPUs
	if (hw_mask) {
		hw_mask->upper = mask >> 32;
		hw_mask->lower = mask;
	} else if (hw_mask_v2) {
		hw_mask_v2->enabled = 1;
		hw_mask_v2->mask[0] = mask;
		hw_mask_v2->mask[1] = mask >> 32;
		hw_mask_v2->mask[2] = mask >> 64;
		hw_mask_v2->mask[3] = mask >> 96;
	} else {
		abort(1, 0, "Stream masking unsupported on this CUDA version (%d), and"
		            " no fallback MASK_OFF set!", ver);
	}
}
}

// currently, one TPC have two SMs
void SMPartitioner::SetGlobalTpcMask(uint64_t mask_64) {
	libsmctrl_set_global_mask(mask_64);
}

void SMPartitioner::SetStreamTpcMask(CUstream s, uint64_t mask_64) {
	auto &stream_last_tpc_mask_map = stream_last_tpc_mask_map_;
	auto it = stream_last_tpc_mask_map.find(s);
	if (it == stream_last_tpc_mask_map.end()) {
		stream_last_tpc_mask_map.insert({s, mask_64});
	} else {
		if (it->second == mask_64) {
			return;
		} else {
			it->second = mask_64;
		}
	}

	// LOG(INFO) << std::hex << "Set Train Stream (" 
  //           << s << ") TPC Mask " << std::hex << mask_64;

	uint128_t mask_128 = -1;
	mask_128 <<= 64;
	mask_128 |= mask_64;
 	libsmctrl_set_stream_mask_ext(s, mask_128);
}


/////////////////////////////////////////////////
namespace {
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
}

std::string SMPartitioner::CheckStreamSM(CUstream s) {
  int* mask = nullptr;
  COL_CUDA_CALL(cudaMallocHost(&mask, 1024 * sizeof(int)));

  get_sm_mask<<<1024, 1, 0, s>>>(mask);
  COL_CUDA_CALL(cudaStreamSynchronize(s));

  std::set<int> used_sms;
  for (int i = 0; i < 1024; i++) {
    used_sms.insert(mask[i]);
  }
  std::stringstream ss;
  ss << "Stream " << s << ", use " << used_sms.size() << " SMs: ";
  for (auto sm : used_sms) {
    ss << sm << " ";
  }

  COL_CUDA_CALL(cudaFreeHost(mask));
  return ss.str();
}

}