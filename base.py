from abc import ABC, abstractmethod
from typing import List, Dict, Any

class InferenceEngine(ABC):
    """Abstract base class for all inference providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Load/initialize the model or create an API client."""

    @abstractmethod
    def infer(self, prompts: List[str], **gen_kwargs) -> List[str]:
        """Core inference interface: takes batched prompts and returns responses."""

    def shutdown(self) -> None:            # Optional: release resources
        pass

class EvaluateEngine(ABC):
    """Abstract base class for all evaluators."""

    def __init__(self, datasetname: str,  **kwargs):
        self.datasetname = datasetname
        self.kwargs = kwargs

    @abstractmethod
    def init_data(self) -> None:
        """Convert input data into prompts."""

    @abstractmethod
    def evaluate(self) -> None:           
        """Evaluate model outputs."""
