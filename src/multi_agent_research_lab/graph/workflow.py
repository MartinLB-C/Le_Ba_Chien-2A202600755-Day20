"""LangGraph workflow skeleton."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

    def build(self) -> object:
        """Create a LangGraph graph."""
        # For simplicity and robustness, returning self as the runnable object
        return self

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        import time
        state.started_at = time.perf_counter()
        
        while not state.completed:
            state = self.supervisor.run(state)
            
            if not state.route_history:
                break
                
            current_route = state.route_history[-1]
            
            if current_route == "researcher":
                state = self.researcher.run(state)
            elif current_route == "analyst":
                state = self.analyst.run(state)
            elif current_route == "writer":
                state = self.writer.run(state)
            elif current_route == "done":
                break
            else:
                state.errors.append(f"Unknown route: {current_route}")
                break
                
        state.finished_at = time.perf_counter()
        return state
