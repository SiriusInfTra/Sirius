import vllm
import torch
from vllm import LLMEngine, EngineArgs

import llm_server

class LLMInference:
    def __init__(self, 
                 model_name: str,
                 max_seq_len: int = None, 
                 max_batch_size: int = None):
        # print("Model Name: ", model_name)
        llm_server.info(f"LLMInfer: Model Name: {model_name}")
        self.model_name = model_name
        engine_args = EngineArgs(
            model=model_name,
            dtype="half",
            max_model_len=None,
            enforce_eager=True,
            enable_chunked_prefill=False,
            use_v2_block_manager=True,
            gpu_memory_utilization=0.5,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)

    def serving_loop(self):
        llm_server.info("LLM Engine Serving Start ...")
        while True:
            if not self.llm_engine.has_unfinished_requests():
                llm_reqs = llm_server.get_llm_requests(1, True)
                print(llm_reqs)
                pass

            request_outputs = self.llm_engine.step()
            # print(request_outputs)

    def process_new_requests(self):
        pass

    def enqueue_infer(self,
                      prompt: str):
        pass
