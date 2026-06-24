
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def test_supervisor_routes_correctly() -> None:
    # We use the MultiAgentWorkflow to test full routing
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    workflow = MultiAgentWorkflow()
    
    # Run the workflow
    result = workflow.run(state)
    
    # Verify the workflow completed successfully
    assert result.completed is True
    
    # Verify the expected routing sequence
    expected_routes = ["researcher", "analyst", "writer", "done"]
    assert result.route_history == expected_routes
    
    # Verify we got an answer
    assert result.final_answer is not None
    assert len(result.final_answer) > 0
    
    # Verify sources were mocked properly
    assert len(result.sources) > 0


def test_search_client_mock() -> None:
    client = SearchClient()
    # Force mock by removing api key from settings if present
    client.settings.tavily_api_key = None
    
    docs = client.search("test query", max_results=2)
    assert len(docs) == 2
    assert docs[0].title == "Introduction to GraphRAG"
