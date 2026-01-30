"""Base agent class with common functionality."""

from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import AgentState, AgentRole, Message
from ..logging_config import get_logger


class BaseAgent(ABC):
    """Base class for all specialized agents."""

    def __init__(
        self,
        llm: BaseChatModel,
        role: AgentRole,
        name: str | None = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            llm: Language model for agent reasoning
            role: Agent role from AgentRole enum
            name: Optional human-readable agent name
        """
        self.llm = llm
        self.role = role
        self.name = name or role.value
        self.logger = get_logger(f"agent.{self.name}")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's role and capabilities."""
        pass

    async def invoke(self, state: AgentState) -> dict[str, Any]:
        """
        Main entry point for agent execution.

        Args:
            state: Current agent state

        Returns:
            Dictionary with state updates
        """
        self.logger.info(
            "agent_invoked",
            agent=self.name,
            stage=state.stage.value,
            workflow_id=state.workflow_id,
        )

        try:
            # Execute agent-specific logic
            result = await self._execute(state)

            # Add agent message to state
            message = Message(
                role=self.role,
                content=result.get("message", "Task completed"),
                metadata=result.get("metadata", {}),
            )

            updates = {
                "messages": [message],
                "current_agent": self.role,
                **result.get("state_updates", {}),
            }

            self.logger.info(
                "agent_completed",
                agent=self.name,
                stage=state.stage.value,
                workflow_id=state.workflow_id,
            )

            return updates

        except Exception as e:
            self.logger.error(
                "agent_failed",
                agent=self.name,
                stage=state.stage.value,
                error=str(e),
                workflow_id=state.workflow_id,
            )

            error_message = Message(
                role=self.role,
                content=f"Error: {str(e)}",
                metadata={"error": True, "exception_type": type(e).__name__},
            )

            return {
                "messages": [error_message],
                "errors": [f"{self.name}: {str(e)}"],
                "current_agent": self.role,
            }

    @abstractmethod
    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the agent's specific logic.

        Args:
            state: Current agent state

        Returns:
            Dictionary containing:
                - message: Agent's response message
                - state_updates: Dictionary of state field updates
                - metadata: Additional metadata
        """
        pass

    async def _call_llm(
        self,
        system_message: str,
        user_message: str,
        temperature: float | None = None,
    ) -> str:
        """
        Call the language model with system and user messages.

        Args:
            system_message: System prompt
            user_message: User query
            temperature: Optional temperature override

        Returns:
            LLM response content
        """
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        if temperature is not None:
            response = await self.llm.ainvoke(messages, temperature=temperature)
        else:
            response = await self.llm.ainvoke(messages)

        return response.content if hasattr(response, "content") else str(response)

    def _extract_previous_context(self, state: AgentState, limit: int = 5) -> str:
        """
        Extract relevant context from previous agent messages.

        Args:
            state: Current agent state
            limit: Maximum number of messages to include

        Returns:
            Formatted string with previous context
        """
        if not state.messages:
            return "No previous context available."

        recent_messages = list(state.messages[-limit:])
        context_parts = []

        for msg in recent_messages:
            context_parts.append(
                f"[{msg.role.value}] ({msg.timestamp.isoformat()}): {msg.content}"
            )

        return "\n".join(context_parts)
