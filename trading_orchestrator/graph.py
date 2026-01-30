"""Main LangGraph orchestration for the trading bot development workflow."""

from typing import Literal, Any
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.language_models import BaseChatModel

from .state import AgentState, AgentRole, WorkflowStage, GraphConfig, Message
from .agents import (
    SupervisorAgent,
    DataEngineerAgent,
    SMCAnalystAgent,
    MLEngineerAgent,
    RiskManagerAgent,
    ValidationAgent,
    SignalEvaluatorAgent,
)
from .logging_config import get_logger

logger = get_logger("graph")


class TradingOrchestrator:
    """
    Main orchestrator for the multi-agent trading bot development workflow.

    This class implements a hierarchical supervisor pattern where a supervisor
    agent coordinates specialized agents to complete the workflow stages.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        config: GraphConfig | None = None,
    ) -> None:
        """
        Initialize the trading orchestrator.

        Args:
            llm: Language model for agent reasoning
            config: Graph configuration (uses defaults if not provided)
        """
        self.llm = llm
        self.config = config or GraphConfig()
        self.logger = get_logger("orchestrator")

        # Initialize all agents
        self.supervisor = SupervisorAgent(llm)
        self.data_engineer = DataEngineerAgent(llm)
        self.smc_analyst = SMCAnalystAgent(llm)
        self.ml_engineer = MLEngineerAgent(llm)
        self.risk_manager = RiskManagerAgent(llm)
        self.validation_agent = ValidationAgent(llm)
        self.signal_evaluator = SignalEvaluatorAgent(llm)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""

        # Create state graph
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("data_engineer", self._data_engineer_node)
        workflow.add_node("smc_analyst", self._smc_analyst_node)
        workflow.add_node("ml_engineer", self._ml_engineer_node)
        workflow.add_node("risk_manager", self._risk_manager_node)
        workflow.add_node("validation_agent", self._validation_agent_node)
        workflow.add_node("signal_evaluator", self._signal_evaluator_node)
        workflow.add_node("human_review", self._human_review_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Add conditional edges from supervisor to route to appropriate agents
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "data_engineer": "data_engineer",
                "smc_analyst": "smc_analyst",
                "ml_engineer": "ml_engineer",
                "risk_manager": "risk_manager",
                "validation_agent": "validation_agent",
                "signal_evaluator": "signal_evaluator",
                "human_review": "human_review",
                "end": END,
            },
        )

        # Each specialist agent returns to supervisor
        for agent_node in [
            "data_engineer",
            "smc_analyst",
            "ml_engineer",
            "risk_manager",
            "validation_agent",
            "signal_evaluator",
        ]:
            workflow.add_edge(agent_node, "supervisor")

        # Human review can go back to supervisor or end
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "supervisor": "supervisor",
                "end": END,
            },
        )

        return workflow

    async def _supervisor_node(self, state: AgentState) -> AgentState:
        """Execute supervisor agent."""
        self.logger.info("supervisor_node_executed", workflow_id=state.workflow_id)
        result = await self.supervisor.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _data_engineer_node(self, state: AgentState) -> AgentState:
        """Execute data engineer agent."""
        self.logger.info("data_engineer_node_executed", workflow_id=state.workflow_id)
        result = await self.data_engineer.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _smc_analyst_node(self, state: AgentState) -> AgentState:
        """Execute SMC analyst agent."""
        self.logger.info("smc_analyst_node_executed", workflow_id=state.workflow_id)
        result = await self.smc_analyst.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _ml_engineer_node(self, state: AgentState) -> AgentState:
        """Execute ML engineer agent."""
        self.logger.info("ml_engineer_node_executed", workflow_id=state.workflow_id)
        result = await self.ml_engineer.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _risk_manager_node(self, state: AgentState) -> AgentState:
        """Execute risk manager agent."""
        self.logger.info("risk_manager_node_executed", workflow_id=state.workflow_id)
        result = await self.risk_manager.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _validation_agent_node(self, state: AgentState) -> AgentState:
        """Execute validation agent."""
        self.logger.info("validation_agent_node_executed", workflow_id=state.workflow_id)
        result = await self.validation_agent.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _signal_evaluator_node(self, state: AgentState) -> AgentState:
        """Execute signal evaluator agent."""
        self.logger.info("signal_evaluator_node_executed", workflow_id=state.workflow_id)
        result = await self.signal_evaluator.invoke(state)
        return AgentState(**{**state.model_dump(), **result})

    async def _human_review_node(self, state: AgentState) -> AgentState:
        """Handle human-in-the-loop review."""
        self.logger.info(
            "human_review_required",
            workflow_id=state.workflow_id,
            stage=state.approval_stage,
        )

        # In production, this would pause and wait for human input
        # For now, we'll simulate approval or provide a hook

        message = Message(
            role=AgentRole.SUPERVISOR,
            content=f"Human review required for stage: {state.approval_stage}. "
                   f"Awaiting approval...",
        )

        return AgentState(
            **{
                **state.model_dump(),
                "messages": [message],
                "stage": WorkflowStage.HUMAN_REVIEW,
            }
        )

    def _route_from_supervisor(
        self,
        state: AgentState,
    ) -> Literal[
        "data_engineer",
        "smc_analyst",
        "ml_engineer",
        "risk_manager",
        "validation_agent",
        "signal_evaluator",
        "human_review",
        "end",
    ]:
        """Route from supervisor to next agent based on state."""

        # Check for completion or failure
        if state.stage in [WorkflowStage.COMPLETED, WorkflowStage.FAILED]:
            self.logger.info(
                "workflow_ending",
                workflow_id=state.workflow_id,
                stage=state.stage.value,
            )
            return "end"

        # Check if human review is required
        if state.requires_human_approval and self.config.enable_human_approval:
            self.logger.info(
                "routing_to_human_review",
                workflow_id=state.workflow_id,
            )
            return "human_review"

        # Route based on next_agent set by supervisor
        if state.next_agent is None:
            self.logger.info(
                "no_next_agent_ending",
                workflow_id=state.workflow_id,
            )
            return "end"

        agent_routing = {
            AgentRole.DATA_ENGINEER: "data_engineer",
            AgentRole.SMC_ANALYST: "smc_analyst",
            AgentRole.ML_ENGINEER: "ml_engineer",
            AgentRole.RISK_MANAGER: "risk_manager",
            AgentRole.VALIDATION_AGENT: "validation_agent",
            AgentRole.SIGNAL_EVALUATOR: "signal_evaluator",
        }

        next_node = agent_routing.get(state.next_agent, "end")

        self.logger.info(
            "routing_decision",
            workflow_id=state.workflow_id,
            next_agent=state.next_agent.value if state.next_agent else None,
            next_node=next_node,
        )

        return next_node  # type: ignore

    def _route_after_human_review(
        self,
        state: AgentState,
    ) -> Literal["supervisor", "end"]:
        """Route after human review."""

        # If human provided feedback, continue to supervisor
        if state.human_feedback:
            self.logger.info(
                "human_feedback_provided",
                workflow_id=state.workflow_id,
            )
            # Reset human review flag
            state.requires_human_approval = False
            return "supervisor"

        # Otherwise, wait for input (in production, this would pause)
        return "end"

    async def compile(self, checkpointer: Any | None = None) -> Any:
        """
        Compile the graph for execution.

        Args:
            checkpointer: Optional checkpointer for state persistence

        Returns:
            Compiled graph ready for execution
        """
        if checkpointer is None and self.config.enable_checkpointing:
            # Create SQLite checkpointer for development
            checkpointer = AsyncSqliteSaver.from_conn_string(
                f"{self.config.checkpoint_dir}/checkpoints.db"
            )

        compiled = self.graph.compile(checkpointer=checkpointer)

        self.logger.info(
            "graph_compiled",
            checkpointing_enabled=self.config.enable_checkpointing,
        )

        return compiled

    def create_initial_state(
        self,
        task_description: str,
        task_parameters: dict[str, Any] | None = None,
    ) -> AgentState:
        """
        Create initial state for workflow execution.

        Args:
            task_description: Description of the trading bot development task
            task_parameters: Additional parameters for the task

        Returns:
            Initial AgentState
        """
        workflow_id = str(uuid.uuid4())

        self.logger.info(
            "workflow_initialized",
            workflow_id=workflow_id,
            task=task_description,
        )

        return AgentState(
            workflow_id=workflow_id,
            stage=WorkflowStage.INITIALIZATION,
            task_description=task_description,
            task_parameters=task_parameters or {},
            started_at=datetime.utcnow(),
            max_iterations=self.config.max_agent_iterations,
        )
