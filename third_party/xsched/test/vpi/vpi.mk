define set_vpi_env
	export LD_LIBRARY_PATH=${LIB_PATH}:$$LD_LIBRARY_PATH; \
	export LD_PRELOAD=${LIB_PATH}/libshimvpi.so:$$LD_PRELOAD
endef

.PHONY: test-pva
test-pva:
	@echo "===== vpi pva preempt test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_pva_preempt ${ASSETS_PATH}/pedestrians.avi ./out.avi; \
	echo "===== vpi pva preempt test complete ====="

.PHONY: test-pva-sa
test-pva-sa:
	@echo "===== vpi pva standalone test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_pva_standalone ${ASSETS_PATH}/pedestrians.avi ./out.avi \
		2> ${RESULTS_PATH}/pva.sa.log; \
	echo "===== vpi pva standalone test complete ====="

.PHONY: test-pva-hpf-base
test-pva-hpf-base:
	@echo "===== vpi pva hpf baseline test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_pva_hpf_base ${ASSETS_PATH}/pedestrians.avi ./out.avi \
		2> ${RESULTS_PATH}/pva.hpf.base.log; \
	echo "===== vpi pva hpf baseline test complete ====="

.PHONY: test-pva-hpf-sched
test-pva-hpf-sched:
	@echo "===== vpi pva hpf sched test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	export XSCHED_POLICY=highest_priority_first; \
	./vpi_pva_hpf_sched ${ASSETS_PATH}/pedestrians.avi ./out.avi \
		2> ${RESULTS_PATH}/pva.hpf.sched.log; \
	echo "===== vpi pva hpf sched test complete ====="

.PHONY: test-pva-cbs-base
test-pva-cbs-base:
	@echo "===== vpi pva cbs baseline test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_pva_cbs_base ${ASSETS_PATH}/pedestrians.avi ./out.avi \
		2> ${RESULTS_PATH}/pva.cbs.base.log; \
	echo "===== vpi pva cbs baseline test complete ====="

.PHONY: test-pva-cbs-sched
test-pva-cbs-sched:
	@echo "===== vpi pva cbs sched test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	export XSCHED_POLICY=constant_bandwidth_server; \
	./vpi_pva_cbs_sched ${ASSETS_PATH}/pedestrians.avi ./out.avi \
		2> ${RESULTS_PATH}/pva.cbs.sched.log; \
	echo "===== vpi pva cbs sched test complete ====="

.PHONY: test-vic
test-vic:
	@echo "===== vpi vic preempt test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_vic_preempt ${ASSETS_PATH}/pedestrians.avi ./out.avi 64 64; \
	echo "===== vpi vic preempt test complete ====="

.PHONY: test-vic-sched
test-vic-sched:
	@echo "===== vpi vic sched test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	export XSCHED_POLICY=${XSCHED_POLICY}; \
	./vpi_vic_sched ${ASSETS_PATH}/pedestrians.avi ./out.avi 64 64; \
	echo "===== vpi vic sched test complete ====="

.PHONY: test-ofa
test-ofa:
	@echo "===== vpi ofa preempt test start ====="; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_ofa_preempt ${ASSETS_PATH}/chair_stereo_left.png ${ASSETS_PATH}/chair_stereo_right.png; \
	echo "===== vpi ofa preempt test complete ====="

.PHONY: test-ofa-sa
test-ofa-sa:
	@echo "===== vpi ofa standalone test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_ofa_standalone ${ASSETS_PATH}/chair_stereo_left.png ${ASSETS_PATH}/chair_stereo_right.png \
		2> ${RESULTS_PATH}/ofa.sa.log; \
	echo "===== vpi ofa standalone test complete ====="

.PHONY: test-ofa-hpf-base
test-ofa-hpf-base:
	@echo "===== vpi ofa hpf baseline test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_ofa_hpf_base ${ASSETS_PATH}/chair_stereo_left.png ${ASSETS_PATH}/chair_stereo_right.png \
		2> ${RESULTS_PATH}/ofa.hpf.base.log; \
	echo "===== vpi ofa hpf baseline test complete ====="

.PHONY: test-ofa-hpf-sched
test-ofa-hpf-sched:
	@echo "===== vpi ofa hpf sched test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	export XSCHED_POLICY=highest_priority_first; \
	./vpi_ofa_hpf_sched ${ASSETS_PATH}/chair_stereo_left.png ${ASSETS_PATH}/chair_stereo_right.png \
		2> ${RESULTS_PATH}/ofa.hpf.sched.log; \
	echo "===== vpi ofa hpf sched test complete ====="

.PHONY: test-ofa-cbs-base
test-ofa-cbs-base:
	@echo "===== vpi ofa cbs baseline test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	./vpi_ofa_cbs_base ${ASSETS_PATH}/chair_stereo_left.png ${ASSETS_PATH}/chair_stereo_right.png \
		2> ${RESULTS_PATH}/ofa.cbs.base.log; \
	echo "===== vpi ofa cbs baseline test complete ====="

.PHONY: test-ofa-cbs-sched
test-ofa-cbs-sched:
	@echo "===== vpi ofa cbs sched test start ====="; \
	mkdir -p ${RESULTS_PATH}; \
	cd ${OUTPUT_PATH}/bin; \
	$(call set_vpi_env); \
	export XSCHED_POLICY=constant_bandwidth_server; \
	./vpi_ofa_cbs_sched ${ASSETS_PATH}/chair_stereo_left.png ${ASSETS_PATH}/chair_stereo_right.png \
		2> ${RESULTS_PATH}/ofa.cbs.sched.log; \
	echo "===== vpi ofa cbs sched test complete ====="
