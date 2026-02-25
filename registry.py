from typing import Dict, Type
from base import InferenceEngine
from base import EvaluateEngine

PROVIDERS: Dict[str, Type[InferenceEngine]] = {}
Evaluators: Dict[str, Type[EvaluateEngine]] = {}

def register(name: str):
    def decorator(cls):
        if not issubclass(cls, InferenceEngine):
            raise ValueError("Provider must inherit InferenceEngine")
        PROVIDERS[name] = cls
        return cls
    return decorator

def register_evaluator(name: str):
    def decorator(cls):
        if not issubclass(cls, EvaluateEngine):
            raise ValueError("Evaluators must inherit EvaluateEngine")
        Evaluators[name] = cls
        return cls
    return decorator