from typing import List
from vllm import LLM, SamplingParams
from base import InferenceEngine
from registry import register

@register("vllm")
class VllmProvider(InferenceEngine):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._setup()

    def _setup(self):
        init_kwargs = self.kwargs  
        model_name = init_kwargs.get("model_name")
        print("model_name:", model_name)

        tensor_parallel_size = init_kwargs.get("tensor_parallel_size", 1)
        dtype = init_kwargs.get("dtype", "bfloat16")
        max_batch_size = init_kwargs.get("max_batch_size", 32)
        gpu_memory_utilization = init_kwargs.get("gpu_memory_utilization", 0.9)
        revision = init_kwargs.get("revision", "main")
        trust_remote_code = init_kwargs.get("trust_remote_code", True)
        load_format = init_kwargs.get("load_format", "auto")
        quantization = init_kwargs.get("quantization", None)

        
        model = model_name

        self.llm = LLM(
            model=model,
            tensor_parallel_size=init_kwargs.get("tensor_parallel_size", 1),
            dtype=init_kwargs.get("dtype", "bfloat16"),
            max_num_batched_tokens=init_kwargs.get("max_batch_size", 40960),  # Increased.
            gpu_memory_utilization=init_kwargs.get("gpu_memory_utilization", 0.9),
            revision=init_kwargs.get("revision", "main"),
            trust_remote_code=init_kwargs.get("trust_remote_code", True),
            load_format=init_kwargs.get("load_format", "auto"),
            quantization=init_kwargs.get("quantization", None),
        )


        gen_kwargs = self.kwargs.get("gen_kwargs", {})
        self.default_params = SamplingParams(
            max_tokens=gen_kwargs.get("max_tokens", 1024),
            temperature=gen_kwargs.get("temperature", 0.7),
            top_p=gen_kwargs.get("top_p", 0.9),
            top_k=gen_kwargs.get("top_k", 40),
            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.05),
        )

    def infer(self, prompts: List[str], **gen_kwargs) -> List[str]:
        params_dict = self.default_params.__dict__.copy()
        params_dict.update(gen_kwargs)
        params = SamplingParams(**params_dict)

        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]
