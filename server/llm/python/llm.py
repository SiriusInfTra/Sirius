from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch
import typing
import logging

class LLMInference:
    def __init__(self, 
                 model_name: str,
                 max_length: int = 512, 
                 max_batch_size: int = None):
        print("Model Name: ", model_name)
        self.model_name = model_name
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.flight_requests = []
        self.queueing_requests = []

    def enqueue_infer(self,
                      prompt: str):
        pass

    def decode(self):
        pass


    

        
        
        