"""
Baseline 1: Lexical & Embedding Similarity Detector.
Measures token overlap, TF-IDF cosine similarity, and semantic vector alignment.
"""
import time
import math
import re
from typing import List, Optional
from src.detectors.base import BaseDetector
from src.core.types import DetectionResult, AtomicClaim
from src.core.claim_extractor import ClaimExtractor

class SimilarityDetector(BaseDetector):
    def __init__(self, threshold: float = 0.45):
        super().__init__(
            name="Similarity Baseline (Lexical + Embedding)",
            description="Computes cosine embedding similarity, token overlap, and claim-level semantic alignment."
        )
        self.threshold = threshold
        self.claim_extractor = ClaimExtractor()

    def predict(
        self,
        prompt: str,
        context: str,
        response: str,
        stochastic_samples: Optional[List[str]] = None
    ) -> DetectionResult:
        start_time = time.time()

        if not context or not context.strip():
            # If no context provided, evaluate response coherence/self-consistency
            context = prompt

        # 1. Lexical Jaccard / Token overlap
        resp_tokens = set(re.findall(r'\w+', response.lower()))
        ctx_tokens = set(re.findall(r'\w+', context.lower()))
        
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'is', 'was', 'were', 'to', 'of', 'and', 'or', 'he', 'she', 'it', 'they', 'this', 'that', 'for', 'with', 'by', 'as', 'be', 'are'}
        resp_content = resp_tokens - stop_words
        ctx_content = ctx_tokens - stop_words

        if not resp_content:
            overlap_ratio = 1.0
        else:
            overlap_ratio = len(resp_content.intersection(ctx_content)) / max(1, len(resp_content))

        # 2. Vector Cosine Similarity (TF-IDF character & word n-grams fallback)
        cosine_sim = self._compute_tfidf_cosine(context, response)

        # Combine into similarity score (higher = more similar to source)
        similarity_score = 0.4 * overlap_ratio + 0.6 * cosine_sim

        # Hallucination score = 1.0 - similarity
        hallucination_score = max(0.0, min(1.0, 1.0 - similarity_score))

        # Extract claims & assign claim scores
        claims = self.claim_extractor.extract_claims(response)
        for claim in claims:
            claim_sim = self._compute_tfidf_cosine(context, claim.text)
            claim.entailment_score = round(claim_sim, 3)
            if claim_sim >= 0.5:
                claim.status = "ENTAILMENT"
            elif claim_sim <= 0.2:
                claim.status = "CONTRADICTION"
            else:
                claim.status = "NEUTRAL"
            claim.confidence = round(claim_sim, 2)

        is_hallucinated = hallucination_score > self.threshold
        elapsed_ms = (time.time() - start_time) * 1000

        # Estimated cost (extremely low: TF-IDF is practically free)
        cost_usd = 0.000001
        flops = len(response.split()) * 500

        return DetectionResult(
            detector_name=self.name,
            hallucination_score=round(float(hallucination_score), 4),
            is_hallucination_predicted=is_hallucinated,
            confidence=round(abs(hallucination_score - self.threshold) * 2, 2),
            latency_ms=round(elapsed_ms, 2),
            estimated_cost_usd=cost_usd,
            estimated_flops=flops,
            atomic_claims=claims,
            metadata={
                "overlap_ratio": round(overlap_ratio, 4),
                "tfidf_cosine": round(cosine_sim, 4),
                "similarity_score": round(similarity_score, 4)
            }
        )

    def _compute_tfidf_cosine(self, doc1: str, doc2: str) -> float:
        """Simple lightweight character/word n-gram cosine similarity."""
        def get_ngram_counts(text: str, n=3):
            text = text.lower()
            counts = {}
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                counts[gram] = counts.get(gram, 0) + 1
            return counts

        v1 = get_ngram_counts(doc1)
        v2 = get_ngram_counts(doc2)

        dot = sum(v1[g] * v2[g] for g in v1 if g in v2)
        norm1 = math.sqrt(sum(val ** 2 for val in v1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in v2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
