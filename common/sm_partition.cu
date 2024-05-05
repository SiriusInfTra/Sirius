#include <common/sm_partition.h>

namespace colserve {

namespace {
// ref: https://github.com/atomicapple0/libsmctrl
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3
static uint64_t g_sm_mask = 0;
static __thread uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;
static void launchCallback(void *ukwn, int domain, int cbid, const void *in_params) {
	if (*(uint32_t*)in_params < 0x50) {
		fprintf(stderr, "Unsupported CUDA version for callback-based SM masking. Aborting...\n");
		return;
	}
	if (!**((uintptr_t***)in_params+8)) {
		fprintf(stderr, "Called with NULL halLaunchDataAllocation\n");
		return;
	}
	//fprintf(stderr, "cta: %lx\n", *(uint64_t*)(**((char***)in_params + 8) + 74));
	// TODO: Check for supported QMD version (>XXX, <4.00)
	// TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 72 bytes (rather than 8 bytes) wide
	uint32_t *lower_ptr = (uint32_t*)(**((char***)in_params + 8) + 84);
	uint32_t *upper_ptr = (uint32_t*)(**((char***)in_params + 8) + 88);

	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}
	fprintf(stderr, "lower mask: %x\n", *lower_ptr);
	fprintf(stderr, "upper mask: %x\n", *upper_ptr);
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
	if (res) {
		fprintf(stderr, "libsmctrl: Error subscribing to launch callback. Error %d\n", res);
		return;
	}
	res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
	if (res)
		fprintf(stderr, "libsmctrl: Error enabling launch callback. Error %d\n", res);
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


// Layout in CUDA's `stream` struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
} __attribute__((packed));

// Should work for CUDA 8.0 through 12.1
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010: {
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
#if __aarch64__
		// Jetson TX2 offset is slightly different on CUDA 9.0.
		// Only compile the check into ARM64 builds.
		int is_parker;
		const char* err_str;
		if ((is_parker = detect_parker_soc()) < 0) {
			cuGetErrorName(-is_parker, &err_str);
			fprintf(stderr, "libsmctrl_set_stream_mask: CUDA call "
					"failed while doing compatibilty test."
			                "Error, '%s'. Not applying stream "
					"mask.\n", err_str);
		}

		if (is_parker)
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF_TX2);
#endif
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
	case 12020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	case 12030:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_3_MASK_OFF);
		break;
	default: {
		// For experimenting to determine the right mask offset, set the MASK_OFF
		// environment variable (positive and negative numbers are supported)
		char* mask_off_str = getenv("MASK_OFF");
		fprintf(stderr, "libsmctrl: Stream masking unsupported on this CUDA version (%d)!\n", ver);
		if (mask_off_str) {
			int off = atoi(mask_off_str);
			fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.1 base %#x "
					"(total off: %#x)\n", off, CU_12_0_MASK_OFF, CU_12_0_MASK_OFF+off);
			hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_12_0_MASK_OFF + off);
		} else {
			return;
		}}
	}

	hw_mask->upper = mask >> 32;
	hw_mask->lower = mask;
}
}




}