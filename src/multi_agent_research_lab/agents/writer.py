"""Writer agent skeleton."""

import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        
        start_time = time.perf_counter()
        
        try:
            system_prompt = f"""You are a technical writer drafting a report for {state.request.audience}.
Synthesize the research and analysis into a clear, comprehensive answer.
Crucially, you MUST include inline citations to the sources provided.
If any claims lack strong evidence, state the limitations clearly."""

            user_prompt = f"Original Query: {state.request.query}\n\nAnalysis Notes:\n{state.analysis_notes}\n\nResearch Notes:\n{state.research_notes}"
            
            response = self.llm.complete(system_prompt, user_prompt)
            state.final_answer = response.content
            
            state.agent_results.append(AgentResult(agent=AgentName.WRITER, content=response.content))
            if response.input_tokens:
                state.token_usage["writer_input"] = state.token_usage.get("writer_input", 0) + response.input_tokens
            if response.output_tokens:
                state.token_usage["writer_output"] = state.token_usage.get("writer_output", 0) + response.output_tokens

            state.add_trace_event("writer.complete", {"duration": time.perf_counter() - start_time})
            
        except Exception as e:
            state.errors.append(f"WriterError: {str(e)}")
            state.add_trace_event("writer.error", {"error": str(e)})

        return state
