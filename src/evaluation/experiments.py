"""
Research Experiment Execution Suite.
Executes the empirical study experiments:
1. Cost vs. Accuracy Trade-Off (Pareto Frontier)
2. Fine-Grained Category Failure Analysis
3. Grounding Context Ablation (No Context vs Partial vs Full RAG Context)
"""
from typing import List, Dict, Any
from src.dataset.benchmark_loader import BenchmarkDataset
from src.evaluation.evaluator import HallucinationEvaluator
from src.detectors.similarity_detector import SimilarityDetector
from src.detectors.nli_detector import NLIDetector
from src.detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from src.detectors.ensemble_detector import EnsembleDetector
from src.core.types import BenchmarkSample, HallucinationCategory

class ExperimentSuite:
    def __init__(self):
        self.dataset = BenchmarkDataset()
        self.evaluator = HallucinationEvaluator()
        self.detectors = [
            SimilarityDetector(),
            NLIDetector(),
            SelfCheckGPTDetector(),
            EnsembleDetector()
        ]

    def run_all_experiments(self) -> Dict[str, Any]:
        samples = self.dataset.get_samples()
        
        # 1. Main Detector Leaderboard
        leaderboard = []
        for det in self.detectors:
            res = self.evaluator.evaluate_detector(det, samples)
            leaderboard.append(res)

        # 2. Experiment 1: Cost vs Accuracy Pareto Analysis
        pareto_analysis = self._run_pareto_analysis(leaderboard)

        # 3. Experiment 2: Hallucination Category Matrix
        category_matrix = self._run_category_matrix(leaderboard)

        # 4. Experiment 3: Context Ablation (RAG vs No RAG)
        context_ablation = self._run_context_ablation(samples)

        return {
            "leaderboard": leaderboard,
            "pareto_analysis": pareto_analysis,
            "category_matrix": category_matrix,
            "context_ablation": context_ablation
        }

    def _run_pareto_analysis(self, leaderboard: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        points = []
        for item in leaderboard:
            points.append({
                "detector": item["detector_name"],
                "accuracy": item["accuracy"],
                "f1": item["f1"],
                "auroc": item["auroc"],
                "cost_usd_1k": item["avg_cost_usd_1k"],
                "latency_ms": item["avg_latency_ms"]
            })
        return points

    def _run_category_matrix(self, leaderboard: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        categories = [c.value for c in HallucinationCategory]
        matrix = {cat: {} for cat in categories}

        for item in leaderboard:
            det_name = item["detector_name"]
            breakdown = item["category_breakdown"]
            for cat in categories:
                if cat in breakdown:
                    matrix[cat][det_name] = breakdown[cat]["accuracy"]
                else:
                    matrix[cat][det_name] = 0.0

        return matrix

    def _run_context_ablation(self, samples: List[BenchmarkSample]) -> Dict[str, Dict[str, float]]:
        """
        Evaluates detectors under 3 context regimes:
        1. No Context (Prompt only)
        2. Partial Context (Truncated/Noisy)
        3. Full Context (Complete RAG)
        """
        ablation_results = {}
        for det in self.detectors:
            # Full context
            full_res = self.evaluator.evaluate_detector(det, samples)

            # Partial context
            partial_samples = [
                BenchmarkSample(
                    id=s.id, prompt=s.prompt, context=s.context[:len(s.context)//2],
                    response=s.response, stochastic_samples=s.stochastic_samples,
                    is_hallucination=s.is_hallucination, category=s.category
                ) for s in samples
            ]
            partial_res = self.evaluator.evaluate_detector(det, partial_samples)

            # No context
            no_ctx_samples = [
                BenchmarkSample(
                    id=s.id, prompt=s.prompt, context="",
                    response=s.response, stochastic_samples=s.stochastic_samples,
                    is_hallucination=s.is_hallucination, category=s.category
                ) for s in samples
            ]
            no_ctx_res = self.evaluator.evaluate_detector(det, no_ctx_samples)

            ablation_results[det.name] = {
                "No Context": no_ctx_res["f1"],
                "Partial Context": partial_res["f1"],
                "Full Context (RAG)": full_res["f1"]
            }

        return ablation_results
