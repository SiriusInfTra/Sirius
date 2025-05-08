SOC_VERSION				= Ascend910B4
ASCEND_MODEL_NAME		= mobilenetv2
ASCEND_MODEL_PATH		= ${WORK_PATH}/test/ascend/models

define set_ascend_env
	export LD_LIBRARY_PATH=${LIB_PATH}:$$LD_LIBRARY_PATH; \
	export LD_PRELOAD=${LIB_PATH}/libshimascend.so:$$LD_PRELOAD
endef

.PHONY: env-ascend
env-ascend:
	@conda create -n ascend python=3.10; \
	conda activate ascend; \
	source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
	conda install pytorch==1.13.1 torchvision cpuonly -c pytorch decorator sympy scipy attrs psutil

.PHONY: ascend-model
ascend-model:
	@mkdir -p ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME}/slices;		\
	cd ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME}/slices;			\
	python3 ${MODEL_SLICE_PATH}/${ASCEND_MODEL_NAME}/onnx.py 64;	\
	mv ${ASCEND_MODEL_NAME}.onnx ../;								\
	bash ${ASCEND_MODEL_PATH}/convert.sh ${SOC_VERSION} .;			\
	cd ..;															\
	echo "converting ${ASCEND_MODEL_NAME}.onnx to 0.om";			\
	atc --model=${ASCEND_MODEL_NAME}.onnx --framework=5 --output=0 --soc_version=${SOC_VERSION}

.PHONY: test-ascend
test-ascend:
	@cd ${OUTPUT_PATH}/bin; \
	$(call set_ascend_env); \
	./ascend_preempt ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME}

.PHONY: test-ascend-sa
test-ascend-sa:
	@cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_ascend_env); \
	./ascend_standalone ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME} \
		2> ${RESULTS_PATH}/${ASCEND_MODEL_NAME}.ascend.sa.log;

.PHONY: test-ascend-hpf-base
test-ascend-hpf-base:
	@cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_ascend_env); \
	./ascend_hpf_base ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME} \
		2> ${RESULTS_PATH}/${ASCEND_MODEL_NAME}.ascend.hpf.base.log;

.PHONY: test-ascend-hpf-sched
test-ascend-hpf-sched:
	@cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_ascend_env); \
	export XSCHED_POLICY=highest_priority_first; \
	./ascend_hpf_sched ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME} \
		2> ${RESULTS_PATH}/${ASCEND_MODEL_NAME}.ascend.hpf.sched.log;

.PHONY: test-ascend-cbs-base
test-ascend-cbs-base:
	@cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_ascend_env); \
	./ascend_cbs_base ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME} \
		2> ${RESULTS_PATH}/${ASCEND_MODEL_NAME}.ascend.cbs.base.log;

.PHONY: test-ascend-cbs-sched
test-ascend-cbs-sched:
	@cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_ascend_env); \
	export XSCHED_POLICY=constant_bandwidth_server; \
	./ascend_cbs_sched ${ASCEND_MODEL_PATH}/${ASCEND_MODEL_NAME} \
		2> ${RESULTS_PATH}/${ASCEND_MODEL_NAME}.ascend.cbs.sched.log;
