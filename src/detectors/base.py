"""
Base Abstract Detector Class for Hallucination Detection.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.core.types import DetectionResult

class BaseDetector(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def predict(
        self,
        prompt: str,
        context: str,
        response: str,
        stochastic_samples: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Evaluate response factuality/hallucination relative to prompt and context.
        Returns unified DetectionResult.
        """
        pass
