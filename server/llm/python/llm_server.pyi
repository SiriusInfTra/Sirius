from typing import List

class LLMRequest:
    request_id: int
    prompt: str
    max_tokens: int

class LLMRequestMetric:
    prompt_tokens: int
    output_tokens: int
    queue_latency_ms: float
    compute_latency_ms: float
    total_latency_ms: float

    def __init__(
        self,
        prompt_tokens: int,
        output_tokens: int,
        queue_latency_ms: float,
        compute_latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        ...

class LLMServer:
    def Init(self) -> None: ...
    def Shutdown(self) -> None: ...
    def GetLLMRequests(self, batch_size: int, timeout_ms: int, block: bool) -> List[LLMRequest]: ...
    def FinishLLMRequest(self, request_id: int, output: str, metric: LLMRequestMetric) -> None: ...
    def IsRunning(self) -> bool: ...

def info(msg: str) -> None: ...
def finish_llm_request(request_id: int, output: str, metric: LLMRequestMetric) -> None: ...
def is_running() -> bool: ...
def check_kv_cache_block_nbytes(
    block_size: int,
    num_layers: int, 
    num_heads: int, 
    head_size: int, 
    block_nbytes: int) -> None: ...