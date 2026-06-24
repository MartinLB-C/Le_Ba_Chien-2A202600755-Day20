"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report", 
        "", 
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Cov | Failure Rate | Notes |", 
        "|---|---:|---:|---:|---:|---:|---|"
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        cov = "" if item.citation_coverage is None else f"{item.citation_coverage:.1%}"
        fail = "" if item.failure_rate is None else f"{item.failure_rate:.1%}"
        
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {cov} | {fail} | {item.notes} |")
        
    # Add failure mode analysis
    lines.append("")
    lines.append("## Failure Modes and Fixes")
    lines.append("- **LLM Hallucination**: Agents may generate facts not in sources. *Fix*: The prompt for the Writer specifically demands inline citations and constraints against making up facts.")
    lines.append("- **Search Quality**: The search might return poor results. *Fix*: We implemented a mock fallback with high-quality sources, and researcher filters duplicates.")
    lines.append("- **Infinite Loops**: The supervisor might route back to the same agent repeatedly. *Fix*: Supervisor enforces a `max_iterations` guard.")
    
    return "\n".join(lines) + "\n"
