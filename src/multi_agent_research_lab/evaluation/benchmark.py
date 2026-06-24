"""Benchmark skeleton for single-agent vs multi-agent."""

import json
import os
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, and quality metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # Calculate Cost
    total_input = sum(v for k, v in state.token_usage.items() if "input" in k)
    total_output = sum(v for k, v in state.token_usage.items() if "output" in k)
    cost = (total_input / 1_000_000 * 0.15) + (total_output / 1_000_000 * 0.6)
    
    # Calculate Quality (simple heuristic based on length and citations for now)
    answer = state.final_answer or ""
    quality = min(10.0, len(answer) / 200.0) # 0 to 10 based on length
    if "error" in answer.lower() or not answer:
        quality = 0.0
        
    # Calculate Citation Coverage
    citations = answer.count("[")
    expected_citations = len(state.sources) if state.sources else 1
    coverage = min(1.0, citations / max(1, expected_citations))
    
    # Failure rate
    failure_rate = 1.0 if len(state.errors) > 0 or not state.final_answer else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name, 
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure_rate,
        notes=f"Iterations: {state.iteration}. Errors: {len(state.errors)}"
    )
    
    # Export trace
    os.makedirs("reports/traces", exist_ok=True)
    with open(f"reports/traces/{run_name}_trace.json", "w") as f:
        json.dump(state.trace, f, indent=2)
        
    return state, metrics
