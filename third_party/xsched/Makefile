# BUILD_TYPE				= Debug / Release
BUILD_TYPE					= Release

# VERBOSE					= ON / OFF : enable verbose makefile
VERBOSE						= OFF

DEVICE_VPI					= OFF

DEVICE_CUDA					= OFF

# CUDA_GEN_CODE				= SM Compute Capability of your CUDA GPU, auto generated
#							  For example, Capability = 8.6, then CUDA_GEN_CODE = 86.
CUDA_GEN_CODE				= 70

DEVICE_CUDLA				= OFF

DEVICE_ASCEND				= OFF

XDAG_SUPPORT				= OFF

WORK_PATH					= $(shell pwd)
TEST_PATH					= ${WORK_PATH}/test
BUILD_PATH					= ${WORK_PATH}/build
OUTPUT_PATH					= ${WORK_PATH}/output
LIB_PATH					= ${OUTPUT_PATH}/lib
RESULTS_PATH				= ${OUTPUT_PATH}/results
ASSETS_PATH					= ${WORK_PATH}/test/assets
MODEL_PATH					= ${WORK_PATH}/test/models
MODEL_SLICE_PATH			= ${WORK_PATH}/tools/slice
XSCHED_POLICY				= highest_priority_first
XSCHED_SERVER_PORT			= 50000

.PHONY: build
build: build/CMakeCache.txt
	rm -rf ${OUTPUT_PATH}
	cmake --build build --target install -- -j$(shell nproc)

build/CMakeCache.txt:
	${MAKE} configure

.PHONY: configure
configure:
	cmake -B${BUILD_PATH}	\
		  -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}	\
		  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}		\
		  -DCMAKE_VERBOSE_MAKEFILE=${VERBOSE}	\
      -DDEVICE_CUDA=${DEVICE_CUDA}			\
		  -DCUDA_GEN_CODE=${CUDA_GEN_CODE}		\
		  -DDEVICE_VPI=${DEVICE_VPI}			\
		  -DDEVICE_CUDLA=${DEVICE_CUDLA}		\
		  -DDEVICE_ASCEND=${DEVICE_ASCEND}		\
		  -DXDAG_SUPPORT=${XDAG_SUPPORT}

.PHONY: clean
clean:
	@rm -rf ${BUILD_PATH} ${OUTPUT_PATH}

.PHONY: vpi
vpi:
	${MAKE} clean; \
	${MAKE} DEVICE_VPI=ON

.PHONY: cuda
cuda:
	${MAKE} clean; \
	${MAKE} DEVICE_CUDA=ON

.PHONY: cudla
cudla:
	${MAKE} clean; \
	${MAKE} DEVICE_CUDLA=ON

.PHONY: ascend
ascend:
	${MAKE} clean; \
	${MAKE} DEVICE_ASCEND=ON

.PHONY: xdag
xdag:
	${MAKE} clean; \
	${MAKE} XDAG_SUPPORT=ON

include service/service.mk
include ${TEST_PATH}/test.mk
