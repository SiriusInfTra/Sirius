CUDLA_MODEL_NAME		= mobilenetv2
CUDLA_MODEL_PATH		= ${WORK_PATH}/test/cudla/models/${CUDLA_MODEL_NAME}
CUDLA_SLICE_PATH		= ${WORK_PATH}/tools/slice/${CUDLA_MODEL_NAME}
CUDLA_ENGINE_PATH		= ${CUDLA_MODEL_PATH}/engines

define set_cudla_env
	export LD_LIBRARY_PATH=${LIB_PATH}:$$LD_LIBRARY_PATH; \
	export LD_PRELOAD=${LIB_PATH}/libshimcudla.so:$$LD_PRELOAD
endef

.PHONY: cudla-model
cudla-model: ${CUDLA_MODEL_PATH}/${CUDLA_MODEL_NAME}.wts
	cd ${CUDLA_MODEL_PATH}; \
	python3 engine.py -s

${CUDLA_MODEL_PATH}/${CUDLA_MODEL_NAME}.wts:
	cd ${CUDLA_MODEL_PATH}; \
	python3 ${CUDLA_SLICE_PATH}/weights.py

.PHONY: test-cudla
test-cudla:
	@echo "===== cudla preempt test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cudla_env); \
	./cudla_preempt ${CUDLA_ENGINE_PATH}; \
	echo "===== cudla preempt test complete ====="

.PHONY: test-cudla-sa
test-cudla-sa:
	@echo "===== cudla standalone test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_cudla_env); \
	./cudla_standalone ${CUDLA_ENGINE_PATH} \
		2> ${RESULTS_PATH}/${CUDLA_MODEL_NAME}.cudla.sa.log; \
	echo "===== cudla bandwidth test complete ====="

.PHONY: test-cudla-hpf-base
test-cudla-hpf-base:
	@echo "===== cudla hpf baseline test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_cudla_env); \
	./cudla_hpf_base ${CUDLA_ENGINE_PATH} \
		2> ${RESULTS_PATH}/${CUDLA_MODEL_NAME}.cudla.hpf.base.log; \
	echo "===== cudla hpf baseline test complete ====="

.PHONY: test-cudla-hpf-sched
test-cudla-hpf-sched:
	@echo "===== cudla hpf sched test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_cudla_env); \
	export XSCHED_POLICY=highest_priority_first; \
	./cudla_hpf_sched ${CUDLA_ENGINE_PATH} \
		2> ${RESULTS_PATH}/${CUDLA_MODEL_NAME}.cudla.hpf.sched.log; \
	echo "===== cudla hpf sched test complete ====="

.PHONY: test-cudla-cbs-base
test-cudla-cbs-base:
	@echo "===== cudla cbs baseline test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_cudla_env); \
	./cudla_cbs_base ${CUDLA_ENGINE_PATH} \
		2> ${RESULTS_PATH}/${CUDLA_MODEL_NAME}.cudla.cbs.base.log; \
	echo "===== cudla cbs baseline test complete ====="

.PHONY: test-cudla-cbs-sched
test-cudla-cbs-sched:
	@echo "===== cudla cbs sched test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	mkdir -p ${RESULTS_PATH}; \
	$(call set_cudla_env); \
	export XSCHED_POLICY=constant_bandwidth_server; \
	./cudla_cbs_sched ${CUDLA_ENGINE_PATH} \
		2> ${RESULTS_PATH}/${CUDLA_MODEL_NAME}.cudla.cbs.sched.log; \
	echo "===== cudla cbs sched test complete ====="
