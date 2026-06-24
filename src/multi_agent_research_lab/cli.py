"""Command-line entrypoint for the lab starter."""

import time
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    console.print(Panel.fit(f"Query: {query}", title="Baseline Started"))
    
    start_time = time.perf_counter()
    llm = LLMClient()
    system_prompt = "You are a helpful assistant. Answer the following query as best as you can."
    response = llm.complete(system_prompt, query)
    
    state.final_answer = response.content
    latency = time.perf_counter() - start_time
    
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline Result"))
    console.print(f"[dim]Latency: {latency:.2f}s | Input Tokens: {response.input_tokens} | Output Tokens: {response.output_tokens}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    
    console.print(Panel.fit(f"Query: {query}", title="Multi-Agent Workflow Started"))
    
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
        
    console.print(Panel.fit(result.final_answer or "No answer produced.", title="Final Answer", style="green"))
    console.print("\n[bold]Route History:[/bold]", " -> ".join(result.route_history))
    
    if result.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"- {err}")
            
    latency = (result.finished_at or 0) - (result.started_at or 0)
    total_input = sum(v for k, v in result.token_usage.items() if "input" in k)
    total_output = sum(v for k, v in result.token_usage.items() if "output" in k)
    console.print(f"\n[dim]Total Latency: {latency:.2f}s | Iterations: {result.iteration} | Input Tokens: {total_input} | Output Tokens: {total_output}[/dim]")


if __name__ == "__main__":
    app()
