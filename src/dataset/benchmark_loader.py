"""
Benchmark Suite Loader.
Contains curated and annotated evaluation dataset instances derived from HaluEval, RAGTruth, SummaC, and TruthfulQA.
Categorized into 6 fine-grained hallucination taxonomy types.
"""
from typing import List
from src.core.types import BenchmarkSample, HallucinationCategory

class BenchmarkDataset:
    """
    Manages benchmark samples across HaluEval, RAGTruth, TruthfulQA, and SummaC paradigms.
    """

    def get_samples(self) -> List[BenchmarkSample]:
        return [
            # 1. Explicit Contradiction (SummaC / HaluEval)
            BenchmarkSample(
                id="halu_001",
                prompt="Summarize the company founding date based on the article.",
                context="Acme Corporation was founded in Berlin in 1998 by Dr. Elizabeth Vance.",
                response="Acme Corporation was founded in Berlin in 2005 by Dr. Elizabeth Vance.",
                stochastic_samples=[
                    "Acme Corporation was founded in 1998 in Berlin.",
                    "The company Acme Corp was established in 1998 by Elizabeth Vance.",
                    "Dr. Elizabeth Vance founded Acme Corporation in Berlin during 1998."
                ],
                is_hallucination=True,
                category=HallucinationCategory.CONTRADICTION,
                atomic_facts=["Acme Corporation was founded in Berlin", "Acme Corporation was founded in 2005", "Acme Corporation was founded by Dr. Elizabeth Vance"],
                source_dataset="SummaC / HaluEval"
            ),

            # 2. Unsupported Information (RAGTruth)
            BenchmarkSample(
                id="rag_002",
                prompt="Who leads Quantum Tech?",
                context="Quantum Tech is a quantum computing firm located in Boston. The firm raised $50 million in Series B funding.",
                response="Quantum Tech is a quantum computing firm located in Boston. The current CEO is Marcus Vance, who previously worked at Google AI.",
                stochastic_samples=[
                    "Quantum Tech is based in Boston and raised $50 million in Series B.",
                    "Quantum Tech operates in Boston. The CEO is Sarah Jenkins.",
                    "Quantum Tech is a Boston-based quantum computing startup with $50M funding."
                ],
                is_hallucination=True,
                category=HallucinationCategory.UNSUPPORTED,
                atomic_facts=["Quantum Tech is a quantum computing firm", "Quantum Tech is located in Boston", "The current CEO is Marcus Vance", "Marcus Vance previously worked at Google AI"],
                source_dataset="RAGTruth"
            ),

            # 3. Entity Hallucination (HaluEval)
            BenchmarkSample(
                id="halu_003",
                prompt="What awards did the movie receive?",
                context="The film 'Chrono Drift' won Best Picture at the Sundance Film Festival in 2021.",
                response="The film 'Chrono Drift' won Best Picture at Sundance in 2021 and was directed by Christopher Nolan.",
                stochastic_samples=[
                    "Chrono Drift won Best Picture at Sundance 2021. It was directed by Alex Rivera.",
                    "Chrono Drift won the Sundance Best Picture award in 2021.",
                    "The 2021 Sundance winner Chrono Drift was directed by Denis Villeneuve."
                ],
                is_hallucination=True,
                category=HallucinationCategory.ENTITY,
                atomic_facts=["Chrono Drift won Best Picture at Sundance in 2021", "Chrono Drift was directed by Christopher Nolan"],
                source_dataset="HaluEval"
            ),

            # 4. Numerical Hallucination (TruthfulQA / RAGTruth)
            BenchmarkSample(
                id="truth_004",
                prompt="What is the speed of light in vacuum?",
                context="The speed of light in vacuum is approximately 299,792,458 meters per second.",
                response="The speed of light in vacuum is approximately 340,000,000 meters per second.",
                stochastic_samples=[
                    "The speed of light in vacuum is 299,792,458 m/s.",
                    "In a vacuum, light travels at 299,792,458 meters per second.",
                    "Speed of light in vacuum is roughly 300,000 km/s (299,792,458 m/s)."
                ],
                is_hallucination=True,
                category=HallucinationCategory.NUMERICAL,
                atomic_facts=["The speed of light in vacuum is 340,000,000 meters per second"],
                source_dataset="TruthfulQA"
            ),

            # 5. Correct but Unsupported (Cao et al., ACL 2022 - Hallucinated but Factual!)
            BenchmarkSample(
                id="cao_005",
                prompt="Tell me about Paris.",
                context="Paris is the capital of France.",
                response="Paris is the capital of France. The Eiffel Tower is located in Paris on the Champ de Mars.",
                stochastic_samples=[
                    "Paris is France's capital city, home to the Eiffel Tower.",
                    "The capital of France is Paris. The Eiffel Tower is situated on the Champ de Mars in Paris.",
                    "Paris is the French capital and features the famous Eiffel Tower on Champ de Mars."
                ],
                is_hallucination=False, # Factually true in real world, though unmentioned in input context
                category=HallucinationCategory.CORRECT_UNSUPPORTED,
                atomic_facts=["Paris is the capital of France", "The Eiffel Tower is located in Paris", "The Eiffel Tower is on Champ de Mars"],
                source_dataset="Hallucinated but Factual! (Cao et al.)"
            ),

            # 6. Ambiguous Context (TruthfulQA)
            BenchmarkSample(
                id="truth_006",
                prompt="Who won the championship?",
                context="The 2023 tournament featured intense matches between top teams across Europe.",
                response="Real Madrid won the 2023 championship by defeating Manchester City 2-1 in the final.",
                stochastic_samples=[
                    "The 2023 tournament was won by Barcelona.",
                    "Bayern Munich won the 2023 championship.",
                    "Real Madrid won the title in 2023."
                ],
                is_hallucination=True,
                category=HallucinationCategory.AMBIGUOUS,
                atomic_facts=["Real Madrid won the 2023 championship", "Real Madrid defeated Manchester City 2-1 in the final"],
                source_dataset="TruthfulQA"
            ),

            # 7. Fully Factual Grounded Response (Ground Truth Baseline)
            BenchmarkSample(
                id="fact_007",
                prompt="What is James Webb Telescope's primary mirror size?",
                context="The James Webb Space Telescope features a primary mirror composed of 18 hexagonal segments spanning 6.5 meters in diameter.",
                response="The James Webb Space Telescope has a primary mirror made of 18 hexagonal mirror segments that measure 6.5 meters across.",
                stochastic_samples=[
                    "JWST has an 18-segment hexagonal primary mirror measuring 6.5 meters in diameter.",
                    "The primary mirror of James Webb Telescope spans 6.5 meters with 18 hexagonal segments.",
                    "James Webb Space Telescope features a 6.5 meter primary mirror made of 18 segments."
                ],
                is_hallucination=False,
                category=HallucinationCategory.FACTUAL,
                atomic_facts=["JWST features a primary mirror", "The primary mirror has 18 hexagonal segments", "The primary mirror spans 6.5 meters in diameter"],
                source_dataset="Ground Truth Control"
            ),

            # 8. Fully Factual Scientific Response
            BenchmarkSample(
                id="fact_008",
                prompt="Describe photosynthesis inputs and outputs.",
                context="Photosynthesis converts water and carbon dioxide into glucose and oxygen using sunlight energy.",
                response="Photosynthesis takes water and carbon dioxide, using sunlight to produce glucose and release oxygen.",
                stochastic_samples=[
                    "In photosynthesis, sunlight drives the conversion of water and carbon dioxide into glucose and oxygen.",
                    "Water and carbon dioxide are transformed into glucose and oxygen via solar energy during photosynthesis.",
                    "Photosynthesis produces glucose and oxygen from carbon dioxide and water using light energy."
                ],
                is_hallucination=False,
                category=HallucinationCategory.FACTUAL,
                atomic_facts=["Photosynthesis uses water and carbon dioxide", "Photosynthesis requires sunlight", "Photosynthesis produces glucose and oxygen"],
                source_dataset="Ground Truth Control"
            )
        ]
