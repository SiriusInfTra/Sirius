.PHONY: run-server
run-server:
	@echo "running schedule server...";	\
	cd ${OUTPUT_PATH}/bin;		\
	export LD_LIBRARY_PATH=${LIB_PATH}:$$LD_LIBRARY_PATH;	\
	./xserver ${XSCHED_POLICY} ${XSCHED_SERVER_PORT}
