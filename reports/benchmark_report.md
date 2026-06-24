# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation Cov | Failure Rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline_1 | 22.57 | 0.0000 | 10.0 | 100.0% | 0.0% | Iterations: 0. Errors: 0 |
| multi_agent_1 | 67.21 | 0.0000 | 10.0 | 100.0% | 0.0% | Iterations: 4. Errors: 0 |
| baseline_2 | 12.26 | 0.0000 | 10.0 | 0.0% | 0.0% | Iterations: 0. Errors: 0 |
| multi_agent_2 | 39.95 | 0.0000 | 10.0 | 100.0% | 0.0% | Iterations: 4. Errors: 0 |
| baseline_3 | 14.31 | 0.0000 | 10.0 | 0.0% | 0.0% | Iterations: 0. Errors: 0 |
| multi_agent_3 | 33.40 | 0.0000 | 10.0 | 100.0% | 0.0% | Iterations: 4. Errors: 0 |

## Failure Modes and Fixes
- **LLM Hallucination**: Agents may generate facts not in sources. *Fix*: The prompt for the Writer specifically demands inline citations and constraints against making up facts.
- **Search Quality**: The search might return poor results. *Fix*: We implemented a mock fallback with high-quality sources, and researcher filters duplicates.
- **Infinite Loops**: The supervisor might route back to the same agent repeatedly. *Fix*: Supervisor enforces a `max_iterations` guard.
