"""Configuration management for the trading orchestrator."""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="anthropic", description="LLM provider to use"
    )
    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="LLM model identifier"
    )
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_max_tokens: int = Field(default=4096, description="Maximum tokens for LLM")

    # Database Configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./trading_orchestrator.db",
        description="Database connection URL"
    )

    # Trading Platform Configuration
    forexsb_api_url: str = Field(
        default="http://localhost:8080",
        description="Forex Strategy Builder API URL"
    )
    forexsb_api_key: str = Field(default="", description="Forex Strategy Builder API key")

    # Agent Configuration
    max_agent_iterations: int = Field(
        default=10,
        description="Maximum iterations per agent"
    )
    agent_timeout_seconds: int = Field(
        default=300,
        description="Timeout for agent operations"
    )
    enable_human_in_loop: bool = Field(
        default=True,
        description="Enable human approval gates"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log format"
    )

    # Monitoring
    enable_metrics: bool = Field(default=False, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")


# Global settings instance
settings = Settings()
