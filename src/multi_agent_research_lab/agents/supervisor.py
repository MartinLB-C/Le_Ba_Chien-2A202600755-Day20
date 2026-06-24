"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        
        # Max iteration guard
        if state.iteration >= self.settings.max_iterations:
            state.add_trace_event("supervisor.route", {"decision": "done", "reason": "max_iterations_reached"})
            state.record_route("done")
            state.completed = True
            return state
            
        # Error guard
        if len(state.errors) > 0:
            state.add_trace_event("supervisor.route", {"decision": "writer", "reason": "error_recovery_to_writer"})
            state.record_route("writer")
            return state

        if not state.sources or not state.research_notes:
            route = "researcher"
            reason = "missing_research"
        elif not state.analysis_notes:
            route = "analyst"
            reason = "missing_analysis"
        elif not state.final_answer:
            route = "writer"
            reason = "missing_final_answer"
        else:
            route = "done"
            reason = "all_steps_completed"
            state.completed = True
            
        state.add_trace_event("supervisor.route", {"decision": route, "reason": reason})
        state.record_route(route)
        
        return state
