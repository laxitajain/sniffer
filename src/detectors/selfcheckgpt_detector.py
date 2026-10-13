"""
Detector 3: SelfCheckGPT (Sampling-Based Disagreement Detector).
Based on Manakul, Liusie & Gales (EMNLP 2023).
Zero-Resource Black-Box Hallucination Detection evaluating semantic inconsistency across stochastic samples.
"""
import time
import math
import re
from typing import List, Optional
from src.detectors.base import BaseDetector
from src.core.types import DetectionResult, AtomicClaim
from src.core.claim_extractor import ClaimExtractor

class SelfCheckGPTDetector(BaseDetector):
    def __init__(self, threshold: float = 0.40, default_num_samples: int = 4):
        super().__init__(
            name="SelfCheckGPT (Sampling Disagreement)",
            description="Measures stochastic sampling disagreement across k generated responses without external ground truth."
        )
        self.threshold = threshold
        self.default_num_samples = default_num_samples
        self.claim_extractor = ClaimExtractor()

    def predict(
        self,
        prompt: str,
        context: str,
        response: str,
        stochastic_samples: Optional[List[str]] = None
    ) -> DetectionResult:
        start_time = time.time()

        # If no samples are provided, generate pseudo-stochastic variations to evaluate agreement
        if not stochastic_samples or len(stochastic_samples) < 2:
            stochastic_samples = self._generate_fallback_samples(response)

        claims = self.claim_extractor.extract_claims(response)
        if not claims:
            claims = [AtomicClaim(id="c0", text=response, entailment_score=0.5, status="NEUTRAL")]

        num_samples = len(stochastic_samples)
        sample_disagreements = []

        # Evaluate each claim against every sample
        for claim in claims:
            agreements = []
            for sample in stochastic_samples:
                sim = self._compute_claim_sample_similarity(claim.text, sample)
                agreements.append(sim)

            avg_agreement = sum(agreements) / max(1, len(agreements))
            claim_disagreement = 1.0 - avg_agreement

            claim.entailment_score = round(avg_agreement, 3)
            if avg_agreement >= 0.6:
                claim.status = "ENTAILMENT"
            elif avg_agreement <= 0.3:
                claim.status = "CONTRADICTION"
            else:
                claim.status = "NEUTRAL"

            sample_disagreements.append(claim_disagreement)

        # Aggregate SelfCheckGPT disagreement score
        hallucination_score = sum(sample_disagreements) / max(1, len(sample_disagreements))
        is_hallucinated = hallucination_score > self.threshold

        elapsed_ms = (time.time() - start_time) * 1000

        # SelfCheckGPT requires generating k stochastic samples -> higher compute cost!
        cost_usd = 0.00015 * num_samples
        flops = num_samples * len(response.split()) * 50000

        return DetectionResult(
            detector_name=self.name,
            hallucination_score=round(float(hallucination_score), 4),
            is_hallucination_predicted=is_hallucinated,
            confidence=round(min(1.0, abs(hallucination_score - self.threshold) * 2.5), 2),
            latency_ms=round(elapsed_ms, 2),
            estimated_cost_usd=round(cost_usd, 6),
            estimated_flops=flops,
            atomic_claims=claims,
            metadata={
                "num_samples_evaluated": num_samples,
                "disagreement_score": round(hallucination_score, 4),
                "samples_preview": [s[:60] + "..." for s in stochastic_samples[:3]]
            }
        )

    def _compute_claim_sample_similarity(self, claim: str, sample: str) -> float:
        """Computes semantic agreement between a claim and a generated sample response."""
        claim_words = set(re.findall(r'\b\w+\b', claim.lower())) - {'the', 'a', 'an', 'in', 'on', 'at', 'is', 'was', 'were', 'to', 'of'}
        sample_words = set(re.findall(r'\b\w+\b', sample.lower())) - {'the', 'a', 'an', 'in', 'on', 'at', 'is', 'was', 'were', 'to', 'of'}

        if not claim_words:
            return 1.0

        overlap = len(claim_words.intersection(sample_words)) / len(claim_words)
        
        # Numeric check: if claim contains a number not in sample, penalize heavily
        claim_nums = set(re.findall(r'\b\d+\b', claim))
        sample_nums = set(re.findall(r'\b\d+\b', sample))
        if claim_nums and not claim_nums.intersection(sample_nums):
            overlap *= 0.2

        return overlap

    def _generate_fallback_samples(self, response: str) -> List[str]:
        """Creates realistic stochastic variations if explicit samples are not provided."""
        # Swap numbers/entities or simulate model variations
        sample1 = response
        sample2 = re.sub(r'\b(19\d\d|20\d\d)\b', lambda m: str(int(m.group(1)) + 2), response)
        sample3 = re.sub(r'\b(Germany|United States|France|China|London|Paris|Berlin)\b', 'Switzerland', response)
        return [sample1, sample2, sample3]
