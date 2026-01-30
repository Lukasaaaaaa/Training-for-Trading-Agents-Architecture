"""LLM factory for creating language model instances."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .config import settings
from .logging_config import get_logger

logger = get_logger("llm_factory")


def create_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Create a language model instance based on configuration.

    Args:
        provider: LLM provider ("openai" or "anthropic"), defaults to config
        model: Model identifier, defaults to config
        temperature: Temperature setting, defaults to config
        max_tokens: Max tokens setting, defaults to config

    Returns:
        Configured language model instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens = max_tokens or settings.llm_max_tokens

    logger.info(
        "creating_llm",
        provider=provider,
        model=model,
        temperature=temperature,
    )

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.openai_api_key,
        )

    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.anthropic_api_key,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_fast_llm() -> BaseChatModel:
    """
    Create a fast, lightweight LLM for simple tasks.

    Returns:
        Configured fast language model
    """
    if settings.llm_provider == "openai":
        model = "gpt-3.5-turbo"
    else:
        model = "claude-3-haiku-20240307"

    return create_llm(model=model, temperature=0.0, max_tokens=2048)


def create_smart_llm() -> BaseChatModel:
    """
    Create a powerful LLM for complex reasoning tasks.

    Returns:
        Configured smart language model
    """
    if settings.llm_provider == "openai":
        model = "gpt-4-turbo-preview"
    else:
        model = "claude-3-5-sonnet-20241022"

    return create_llm(model=model, temperature=0.1, max_tokens=4096)
