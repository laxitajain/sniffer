"""
Evaluation Metric Suite for Hallucination Detectors.
Computes Accuracy, Precision, Recall, F1, AUROC, Latency, Cost, and Category-level breakdown.
"""
import numpy as np
from typing import List, Dict, Any
from src.core.types import BenchmarkSample, DetectionResult, DetectorMetrics, HallucinationCategory
from src.detectors.base import BaseDetector

class HallucinationEvaluator:
    def evaluate_detector(self, detector: BaseDetector, samples: List[BenchmarkSample]) -> Dict[str, Any]:
        """
        Runs a detector on all benchmark samples and computes metrics.
        """
        y_true = []
        y_pred = []
        y_scores = []

        latencies = []
        costs = []

        category_results: Dict[str, Dict[str, List]] = {}

        for s in samples:
            res: DetectionResult = detector.predict(
                prompt=s.prompt,
                context=s.context,
                response=s.response,
                stochastic_samples=s.stochastic_samples
            )

            is_true_hallu = s.is_hallucination
            is_pred_hallu = res.is_hallucination_predicted
            score = res.hallucination_score

            y_true.append(1 if is_true_hallu else 0)
            y_pred.append(1 if is_pred_hallu else 0)
            y_scores.append(score)

            latencies.append(res.latency_ms)
            costs.append(res.estimated_cost_usd)

            cat_key = s.category.value if isinstance(s.category, HallucinationCategory) else str(s.category)
            if cat_key not in category_results:
                category_results[cat_key] = {"true": [], "pred": [], "scores": []}
            
            category_results[cat_key]["true"].append(1 if is_true_hallu else 0)
            category_results[cat_key]["pred"].append(1 if is_pred_hallu else 0)
            category_results[cat_key]["scores"].append(score)

        # Compute standard classification metrics
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        y_s = np.array(y_scores)

        tp = np.sum((y_t == 1) & (y_p == 1))
        fp = np.sum((y_t == 0) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        tn = np.sum((y_t == 0) & (y_p == 0))

        accuracy = float((tp + tn) / max(1, len(y_t)))
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = float(2 * precision * recall / max(1e-6, precision + recall))

        auroc = self._compute_auroc(y_t, y_s)
        avg_latency = float(np.mean(latencies))
        avg_cost = float(np.mean(costs))

        # Compute category-wise accuracy
        category_breakdown = {}
        for cat, data in category_results.items():
            c_t = np.array(data["true"])
            c_p = np.array(data["pred"])
            c_acc = float(np.sum(c_t == c_p) / max(1, len(c_t)))
            category_breakdown[cat] = {
                "accuracy": round(c_acc, 4),
                "count": len(c_t),
                "avg_score": round(float(np.mean(data["scores"])), 4)
            }

        return {
            "detector_name": detector.name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auroc": round(auroc, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_cost_usd_1k": round(avg_cost * 1000, 4),
            "category_breakdown": category_breakdown
        }

    def _compute_auroc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Computes AUROC using rank-sum statistic (Mann-Whitney U)."""
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Pairwise comparison
        pos_scores = y_scores[pos_mask]
        neg_scores = y_scores[neg_mask]

        greater = 0
        ties = 0

        for p in pos_scores:
            for n in neg_scores:
                if p > n:
                    greater += 1
                elif p == n:
                    ties += 0.5

        return float((greater + ties) / (n_pos * n_neg))
