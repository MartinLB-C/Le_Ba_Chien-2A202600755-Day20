import os
import yaml
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.logging import configure_logging

def run_baseline(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    llm = LLMClient()
    system_prompt = "You are a helpful assistant. Answer the following query as best as you can."
    response = llm.complete(system_prompt, query)
    state.final_answer = response.content
    if response.input_tokens:
        state.token_usage["baseline_input"] = response.input_tokens
    if response.output_tokens:
        state.token_usage["baseline_output"] = response.output_tokens
    return state

def run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)

def main():
    settings = get_settings()
    configure_logging(settings.log_level)
    
    queries = []
    try:
        with open("configs/lab_default.yaml") as f:
            config = yaml.safe_load(f)
            queries = config.get("queries", [])[:3]
    except Exception:
        pass
        
    if not queries:
        queries = ["Research GraphRAG state-of-the-art", "Explain multi-agent systems", "Guardrails in generative AI"]
        
    metrics = []
    
    for i, q in enumerate(queries):
        query_text = q.get("query", q) if isinstance(q, dict) else q
        
        print(f"Running baseline {i+1}...")
        _, b_metric = run_benchmark(f"baseline_{i+1}", query_text, run_baseline)
        metrics.append(b_metric)
        
        print(f"Running multi-agent {i+1}...")
        _, ma_metric = run_benchmark(f"multi_agent_{i+1}", query_text, run_multi_agent)
        metrics.append(ma_metric)
        
    report_md = render_markdown_report(metrics)
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_report.md", "w") as f:
        f.write(report_md)
        
    print("Benchmark complete. Report saved to reports/benchmark_report.md")

if __name__ == "__main__":
    main()
