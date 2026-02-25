from typing import List
from openai import OpenAI
from base import InferenceEngine
from registry import register

@register("qwen3")
class Qwen3Provider(InferenceEngine):
    def _setup(self):
        self.client = OpenAI(base_url=self.kwargs.get("base_url"),
                             api_key=self.kwargs["api_key"])
        self.model = self.model_name

    def infer(self, prompts: List[str], **gen_kwargs) -> List[str]:
        responses = []
        for prompt in prompts:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **gen_kwargs
            )
            answer_content=''
            reasoning_content=''
            for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content

                
                if hasattr(delta, "content") and delta.content:
                    answer_content += delta.content

            responses.append(f"reasoning_content:{reasoning_content}\nanswer_content:{answer_content}")
        return responses
