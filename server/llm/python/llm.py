from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import typing
import logging
import pathlib


class LLMWorker:
    def __init__(self, 
                 model_name: str, 
                 max_seq_len: int):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # self.model = self.model.cuda()
        self._init_infer()

    def _init_infer(self):
        pass

    def decode(self):
        pass
        

class LLMInference:
    def __init__(self, 
                 model_name: str,
                 max_seq_len: int = 512, 
                 max_batch_size: int = None):
        print("Model Name: ", model_name)
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.flight_requests = []
        self.queueing_requests = []
        self.llm_worker = LLMWorker(model_name, max_seq_len)

    def enqueue_infer(self,
                      prompt: str):
        pass
