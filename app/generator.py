"""Generator interface for rag-local-eval-loop."""
from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Answer:
    text: str
    grounded: bool
    generation_ms: float
    model: str = "OVX-1-Extractive"

def generate_answer(query: str, results: list) -> Answer:
    t0 = time.perf_counter()
    if not results:
        t1 = time.perf_counter()
        return Answer(
            text="I don't have enough context to answer this question.",
            grounded=False,
            generation_ms=(t1 - t0) * 1000,
        )

    top_text = results[0].text if results else ""
    t1 = time.perf_counter()
    return Answer(
        text=top_text,
        grounded=True,
        generation_ms=(t1 - t0) * 1000,
    )
