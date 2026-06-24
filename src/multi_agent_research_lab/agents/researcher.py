"""Researcher agent skeleton."""

import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        
        start_time = time.perf_counter()
        
        try:
            # 1. Search
            docs = self.search.search(state.request.query, max_results=state.request.max_sources)
            
            # Deduplicate by URL
            seen_urls = set()
            unique_docs = []
            for d in docs:
                if d.url and d.url not in seen_urls:
                    seen_urls.add(d.url)
                    unique_docs.append(d)
                elif not d.url:
                    unique_docs.append(d)
                    
            state.sources = unique_docs
            
            # 2. Summarize notes using LLM
            system_prompt = "You are a research assistant. Extract key facts from the provided sources as bullet points. Always include the source title/URL in brackets []."
            
            sources_text = "\n\n".join([f"Source: {d.title}\nURL: {d.url}\nContent: {d.snippet}" for d in unique_docs])
            user_prompt = f"Query: {state.request.query}\n\nSources:\n{sources_text}"
            
            response = self.llm.complete(system_prompt, user_prompt)
            state.research_notes = response.content
            
            # Record metrics
            state.agent_results.append(AgentResult(agent=AgentName.RESEARCHER, content=response.content))
            if response.input_tokens:
                state.token_usage["researcher_input"] = state.token_usage.get("researcher_input", 0) + response.input_tokens
            if response.output_tokens:
                state.token_usage["researcher_output"] = state.token_usage.get("researcher_output", 0) + response.output_tokens
                
            state.add_trace_event("researcher.complete", {
                "num_sources": len(unique_docs), 
                "duration": time.perf_counter() - start_time
            })
            
        except Exception as e:
            state.errors.append(f"ResearcherError: {str(e)}")
            state.add_trace_event("researcher.error", {"error": str(e)})

        return state
