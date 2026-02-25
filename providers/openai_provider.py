from typing import List
from openai import OpenAI
from base import InferenceEngine
from registry import register

@register("openai")
class OpenAIProvider(InferenceEngine):
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
            responses.append(resp.choices[0].message.content)
        return responses
