"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

try:
    import openai  # type: ignore
except ImportError:
    openai = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        if self.settings.llm_api_key and openai is not None:
            kwargs = {"api_key": self.settings.llm_api_key}
            if self.settings.llm_base_url:
                kwargs["base_url"] = self.settings.llm_base_url
            self.client = openai.OpenAI(**kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        
        if self.client is None:
            # Fallback for offline/test mode without API key
            return LLMResponse(
                content="[Mock LLM Response] I am a standalone fallback model. To get real responses, set LLM_API_KEY.",
                input_tokens=10,
                output_tokens=20,
                cost_usd=0.0
            )

        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        if response.choices:
            content = response.choices[0].message.content or ""
        else:
            # Handle non-standard responses (e.g. Aliyun DashScope) which may return text directly
            content = getattr(response, "text", None)
            if content is None and hasattr(response, "model_extra") and response.model_extra:
                content = response.model_extra.get("text", "")
            content = content or ""
            
        usage = response.usage
        
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        # Simple cost estimation (e.g., using generic GPT-4o-mini pricing)
        # $0.150 / 1M input tokens, $0.600 / 1M output tokens
        cost = (input_tokens / 1_000_000 * 0.15) + (output_tokens / 1_000_000 * 0.6)
        
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost
        )
