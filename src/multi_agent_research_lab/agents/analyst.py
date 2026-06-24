"""Analyst agent skeleton."""

import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        
        start_time = time.perf_counter()
        
        try:
            system_prompt = """You are an analyst. Review the research notes and extract:
1. Key claims
2. Supporting evidence
3. Weak/uncertain claims or contradictory viewpoints
4. Comparison points

Output a structured markdown analysis."""

            user_prompt = f"Research Notes:\n{state.research_notes}"
            
            response = self.llm.complete(system_prompt, user_prompt)
            state.analysis_notes = response.content
            
            state.agent_results.append(AgentResult(agent=AgentName.ANALYST, content=response.content))
            if response.input_tokens:
                state.token_usage["analyst_input"] = state.token_usage.get("analyst_input", 0) + response.input_tokens
            if response.output_tokens:
                state.token_usage["analyst_output"] = state.token_usage.get("analyst_output", 0) + response.output_tokens

            state.add_trace_event("analyst.complete", {"duration": time.perf_counter() - start_time})
            
        except Exception as e:
            state.errors.append(f"AnalystError: {str(e)}")
            state.add_trace_event("analyst.error", {"error": str(e)})

        return state
