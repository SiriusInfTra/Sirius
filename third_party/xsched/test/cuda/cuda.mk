BATCH_SIZE					= 1
TRAIN_BATCH_SIZE			= 64
MODEL						= resnet152
MODELS						= mobilenetv1 resnet50 resnet152 vgg19 deepsort inceptionv3 yolov5l densenet121 densenet201

define set_cuda_env
	export LD_LIBRARY_PATH=${LIB_PATH}:$$LD_LIBRARY_PATH
endef

.PHONY: test-cuda-kill
test-cuda-kill:
	@cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	./cuda_kill

.PHONY: test-trt
test-trt:
	@echo "===== tensorrt inference test for ${MODEL} start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	echo "log: ${RESULTS_PATH}/${MODEL}.trt.log"; \
	./trt_preempt ${MODEL_PATH} ./data ${MODEL} ${BATCH_SIZE} ${RESULTS_PATH} \
		2> ${RESULTS_PATH}/${MODEL}.trt.log; \
	echo "===== tensorrt inference test for ${MODEL} complete ====="

.PHONY: test-trt-sa
test-trt-sa:
	@echo "===== tensorrt inference standalone test for ${MODEL} start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	echo "log: ${RESULTS_PATH}/${MODEL}.trt.sa.log"; \
	./trt_standalone ${MODEL_PATH} ./data ${MODEL} ${BATCH_SIZE} ${RESULTS_PATH} \
		2> ${RESULTS_PATH}/${MODEL}.trt.sa.log; \
	echo "===== tensorrt inference bandwidth test for ${MODEL} complete ====="

.PHONY: test-trt-hpf-base
test-trt-hpf-base:
	@echo "===== tensorrt hpf baseline test for ${MODEL} start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	echo "log: ${RESULTS_PATH}/${MODEL}.trt.hpf.base.log"; \
	./trt_hpf_base ${MODEL_PATH} ./data ${MODEL} ${BATCH_SIZE} ${RESULTS_PATH} \
		2> ${RESULTS_PATH}/${MODEL}.trt.hpf.base.log; \
	echo "===== tensorrt hpf baseline test for ${MODEL} complete ====="

.PHONY: test-trt-hpf-sched
test-trt-hpf-sched:
	@echo "===== tensorrt hpf sched test for ${MODEL} start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	export XSCHED_POLICY=highest_priority_first; \
	echo "log: ${RESULTS_PATH}/${MODEL}.trt.hpf.sched.log"; \
	./trt_hpf_sched ${MODEL_PATH} ./data ${MODEL} ${BATCH_SIZE} ${RESULTS_PATH} \
		2> ${RESULTS_PATH}/${MODEL}.trt.hpf.sched.log; \
	echo "===== tensorrt hpf sched test for ${MODEL} complete ====="

.PHONY: test-trt-cbs-base
test-trt-cbs-base:
	@echo "===== tensorrt cbs baseline test for ${MODEL} start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	echo "log: ${RESULTS_PATH}/${MODEL}.trt.cbs.base.log"; \
	./trt_cbs_base ${MODEL_PATH} ./data ${MODEL} ${BATCH_SIZE} ${RESULTS_PATH} \
		2> ${RESULTS_PATH}/${MODEL}.trt.cbs.base.log; \
	echo "===== tensorrt cbs baseline test for ${MODEL} complete ====="

.PHONY: test-trt-cbs-sched
test-trt-cbs-sched:
	@echo "===== tensorrt cbs sched test for ${MODEL} start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_cuda_env); \
	export XSCHED_POLICY=constant_bandwidth_server; \
	echo "log: ${RESULTS_PATH}/${MODEL}.trt.cbs.sched.log"; \
	./trt_cbs_sched ${MODEL_PATH} ./data ${MODEL} ${BATCH_SIZE} ${RESULTS_PATH} \
		2> ${RESULTS_PATH}/${MODEL}.trt.cbs.sched.log; \
	echo "===== tensorrt cbs sched test for ${MODEL} complete ====="

.PHONY: test-trt-all
test-trt-all:
	@for model in ${MODELS}; do \
		make test-trt MODEL=$${model}; \
	done

.PHONY: env-cuda
env-cuda:
	docker run						\
		-it							\
		--name xsched-cuda			\
		--gpus all					\
		--privileged=true			\
		-v ${PWD}:/xproject/xsched	\
		shenwhang/xsched-cuda:0.3	\
		/bin/bash
