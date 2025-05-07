import vllm
import torch
import os
from typing import Optional
from vllm import LLMEngine, EngineArgs, SamplingParams

import llm_server

class LLMInference:
    def __init__(self, 
                 model_name: str,
                 max_model_len: Optional[int] = None,
                 max_seq_len: Optional[int] = None):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.max_seq_len = max_seq_len
        self.stream = torch.cuda.Stream()
        llm_server.info(f'LLMInfer: Stream {self.stream.cuda_stream}')
        torch.cuda.set_stream(self.stream)
        self._process_args()

        # print("Model Name: ", model_name)
        # if os.environ['https_proxy'] is None:
        # if 'https_proxy' not in os.environ:
        #     os.environ['https_proxy'] = 'http://127.0.0.1:7890'
        llm_server.info(f"LLMInfer: Model Name: {self.model_name}")
        engine_args = EngineArgs(
            model=self.model_name,
            dtype="half",
            max_model_len=self.max_model_len,
            enforce_eager=True,
            enable_chunked_prefill=False,
            use_v2_block_manager=True,
            gpu_memory_utilization=0.95,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)

        # del os.environ['https_proxy']

    def _process_args(self):
        if self.max_model_len == 0:
            self.max_model_len = None
        if self.max_seq_len == 0:
            self.max_seq_len = None

        llm_server.info_with_frame(
            f'LLMInfer Args: \n'
            f'\tModel Name: {self.model_name}\n'
            f'\tmax_model_len {self.max_model_len}'
            f' | max_seq_len {self.max_seq_len}')

    def serving_loop(self):
        torch.cuda.set_stream(self.stream)
        llm_server.info_with_frame(
            f"LLM Engine Serving Start, current stream {torch.cuda.current_stream().cuda_stream}")
        while llm_server.is_running() or self.llm_engine.has_unfinished_requests():
            if not self.llm_engine.has_unfinished_requests():
                llm_reqs = llm_server.get_llm_requests(1, 10, True)
                if len(llm_reqs) == 0:
                    break
                self.process_new_requests(llm_reqs)
                
            llm_reqs = llm_server.get_llm_requests(1, 0, False)
            if len(llm_reqs) > 0:
                self.process_new_requests(llm_reqs)
    
            request_outputs = self.llm_engine.step()
            for req_out in request_outputs:
                if req_out.finished:
                    metric = llm_server.LLMRequestMetric(
                        len(req_out.prompt_token_ids),
                        len(req_out.outputs[0].token_ids),
                        (req_out.metrics.first_scheduled_time - req_out.metrics.arrival_time) * 1000,
                        (req_out.metrics.first_token_time - req_out.metrics.first_scheduled_time) * 1000,
                        (req_out.metrics.last_token_time - req_out.metrics.first_token_time) * 1000,
                    )
                    # llm_server.info(f'metric: {metric} | {(req_out.metrics.finished_time - req_out.metrics.last_token_time) * 1000}')
                    llm_server.finish_llm_request(
                        req_out.request_id, 
                        req_out.outputs[0].text,
                        metric   
                    )
                    
            # print(request_outputs)
        llm_server.info_with_frame("LLM Engine Serving End ...")

    def process_new_requests(self, llm_reqs):
        for req in llm_reqs:
            self.llm_engine.add_request(
                request_id=req.request_id, 
                prompt=req.prompt, 
                sampling_params=SamplingParams(
                    temperature=0.8, top_p=0.95,
                    max_tokens=req.max_tokens,
                )
            )

    def get_exec_stream(self):
        return self.stream.cuda_stream

    def enqueue_infer(self,
                      prompt: str):
        pass
