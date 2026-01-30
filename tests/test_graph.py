"""Tests for graph orchestration."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from trading_orchestrator.graph import TradingOrchestrator
from trading_orchestrator.state import GraphConfig, AgentState, WorkflowStage


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
    return llm


@pytest.fixture
def orchestrator(mock_llm):
    """Create an orchestrator instance for testing."""
    config = GraphConfig(
        enable_checkpointing=False,
        enable_human_approval=False,
    )
    return TradingOrchestrator(mock_llm, config)


def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization."""
    assert orchestrator.supervisor is not None
    assert orchestrator.data_engineer is not None
    assert orchestrator.smc_analyst is not None
    assert orchestrator.ml_engineer is not None
    assert orchestrator.risk_manager is not None
    assert orchestrator.validation_agent is not None
    assert orchestrator.signal_evaluator is not None
    assert orchestrator.graph is not None


def test_create_initial_state(orchestrator):
    """Test initial state creation."""
    task = "Test trading bot development"
    params = {"symbol": "EURUSD", "timeframe": "H1"}

    state = orchestrator.create_initial_state(task, params)

    assert state.workflow_id is not None
    assert state.task_description == task
    assert state.task_parameters == params
    assert state.stage == WorkflowStage.INITIALIZATION
    assert state.iteration == 0


@pytest.mark.asyncio
async def test_compile_graph(orchestrator):
    """Test graph compilation."""
    compiled = await orchestrator.compile()
    assert compiled is not None


def test_route_from_supervisor_data_engineer(orchestrator):
    """Test routing to data engineer."""
    state = AgentState(
        workflow_id="test-001",
        stage=WorkflowStage.DATA_PREPARATION,
        next_agent="data_engineer",
        task_description="Test",
    )

    # Note: next_agent is a string in the state but AgentRole enum is expected
    from trading_orchestrator.state import AgentRole
    state.next_agent = AgentRole.DATA_ENGINEER

    route = orchestrator._route_from_supervisor(state)
    assert route == "data_engineer"


def test_route_from_supervisor_end(orchestrator):
    """Test routing to end when completed."""
    state = AgentState(
        workflow_id="test-001",
        stage=WorkflowStage.COMPLETED,
        task_description="Test",
    )

    route = orchestrator._route_from_supervisor(state)
    assert route == "end"


def test_route_from_supervisor_human_review(orchestrator):
    """Test routing to human review."""
    # Enable human approval for this test
    orchestrator.config.enable_human_approval = True

    state = AgentState(
        workflow_id="test-001",
        stage=WorkflowStage.VALIDATION,
        requires_human_approval=True,
        task_description="Test",
    )

    route = orchestrator._route_from_supervisor(state)
    assert route == "human_review"


def test_route_after_human_review_with_feedback(orchestrator):
    """Test routing after human review with feedback."""
    state = AgentState(
        workflow_id="test-001",
        stage=WorkflowStage.HUMAN_REVIEW,
        human_feedback="Approved",
        task_description="Test",
    )

    route = orchestrator._route_after_human_review(state)
    assert route == "supervisor"


def test_route_after_human_review_without_feedback(orchestrator):
    """Test routing after human review without feedback."""
    state = AgentState(
        workflow_id="test-001",
        stage=WorkflowStage.HUMAN_REVIEW,
        human_feedback=None,
        task_description="Test",
    )

    route = orchestrator._route_after_human_review(state)
    assert route == "end"
