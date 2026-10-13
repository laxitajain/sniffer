"""
Core data types and definitions for Hallucination Detection Empirical Study.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

class HallucinationCategory(str, Enum):
    CONTRADICTION = "Explicit Contradiction"
    UNSUPPORTED = "Unsupported Information"
    ENTITY = "Entity Hallucination"
    NUMERICAL = "Numerical Hallucination"
    CORRECT_UNSUPPORTED = "Correct but Unsupported"
    AMBIGUOUS = "Ambiguous Context"
    FACTUAL = "Factual / Correct"

@dataclass
class AtomicClaim:
    id: str
    text: str
    entailment_score: float = 0.0
    status: str = "NEUTRAL" # ENTAILMENT, CONTRADICTION, NEUTRAL
    supporting_context_span: Optional[str] = None
    confidence: float = 0.0

@dataclass
class BenchmarkSample:
    id: str
    prompt: str
    context: str
    response: str
    stochastic_samples: List[str] = field(default_factory=list)
    is_hallucination: bool = False
    category: HallucinationCategory = HallucinationCategory.FACTUAL
    atomic_facts: List[str] = field(default_factory=list)
    source_dataset: str = "Synthetic/Benchmark"

@dataclass
class DetectionResult:
    detector_name: str
    hallucination_score: float  # 0.0 (fully factual) to 1.0 (definitely hallucinated)
    is_hallucination_predicted: bool
    confidence: float
    latency_ms: float
    estimated_cost_usd: float
    estimated_flops: int
    atomic_claims: List[AtomicClaim] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectorMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    avg_latency_ms: float
    avg_cost_usd: float
    category_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
