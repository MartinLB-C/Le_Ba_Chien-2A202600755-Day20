"""Search client abstraction for ResearcherAgent."""

import json
import urllib.request

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        
        if self.settings.tavily_api_key:
            return self._search_tavily(query, max_results)
        
        return self._search_mock(query, max_results)
        
    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        url = "https://api.tavily.com/search"
        data = json.dumps({
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                
            docs = []
            for r in result.get("results", []):
                docs.append(SourceDocument(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", "")[:500],
                    metadata={"score": r.get("score", 0)}
                ))
            return docs
        except Exception as e:
            # Fallback to mock on error
            print(f"Tavily search failed: {e}. Falling back to mock.")
            return self._search_mock(query, max_results)

    def _search_mock(self, query: str, max_results: int) -> list[SourceDocument]:
        """Fallback mock returning high-quality generic AI sources."""
        mocks = [
            SourceDocument(
                title="Introduction to GraphRAG",
                url="https://example.com/graphrag-intro",
                snippet="GraphRAG (Graph-based Retrieval-Augmented Generation) combines knowledge graphs with LLMs to improve context retrieval, particularly for global questions over large document sets.",
                metadata={"source": "mock"}
            ),
            SourceDocument(
                title="Evaluating Multi-Agent Systems",
                url="https://example.com/eval-multi-agent",
                snippet="Multi-agent evaluation requires measuring not just the final output quality, but also the trace history, inter-agent communication, and routing decisions to ensure reliable performance.",
                metadata={"source": "mock"}
            ),
            SourceDocument(
                title="LangGraph vs AutoGen",
                url="https://example.com/langgraph-autogen",
                snippet="LangGraph structures agent workflows as directed graphs with explicit state management, while AutoGen relies more on conversational turns and prompt-based orchestration between agents.",
                metadata={"source": "mock"}
            ),
            SourceDocument(
                title="Guardrails in Generative AI",
                url="https://example.com/ai-guardrails",
                snippet="Implementing guardrails for LLMs involves structural validation, output parsing, and intermediate steps where an agent critically reviews the generated content before exposing it.",
                metadata={"source": "mock"}
            ),
            SourceDocument(
                title="The Role of Supervisor Agents",
                url="https://example.com/supervisor-agents",
                snippet="A supervisor agent acts as a router that determines the next step in a workflow based on the current state. It helps prevent infinite loops and ensures sub-tasks are completed.",
                metadata={"source": "mock"}
            )
        ]
        return mocks[:max_results]
