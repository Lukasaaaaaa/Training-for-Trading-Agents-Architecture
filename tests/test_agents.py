"""Tests for specialized agents."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from trading_orchestrator.agents import (
    DataEngineerAgent,
    SMCAnalystAgent,
    MLEngineerAgent,
    RiskManagerAgent,
    ValidationAgent,
    SignalEvaluatorAgent,
)
from trading_orchestrator.state import AgentState, DatasetInfo, WorkflowStage


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
    return llm


@pytest.fixture
def base_state():
    """Create a base state for testing."""
    return AgentState(
        workflow_id="test-001",
        stage=WorkflowStage.INITIALIZATION,
        task_description="Test workflow",
        task_parameters={"symbol": "EURUSD", "timeframe": "H1"},
    )


@pytest.mark.asyncio
async def test_data_engineer_prepare_dataset(mock_llm, base_state):
    """Test data engineer dataset preparation."""
    agent = DataEngineerAgent(mock_llm)

    result = await agent.invoke(base_state)

    assert "dataset" in result
    assert result["dataset"].symbol == "EURUSD"
    assert result["dataset"].timeframe == "H1"
    assert result["dataset"].total_bars > 0
    assert 0 <= result["dataset"].quality_score <= 1.0


@pytest.mark.asyncio
async def test_smc_analyst_feature_generation(mock_llm, base_state):
    """Test SMC analyst feature generation."""
    # Add dataset to state
    base_state.dataset = DatasetInfo(
        symbol="EURUSD",
        timeframe="H1",
        start_date=datetime.utcnow() - timedelta(days=730),
        end_date=datetime.utcnow(),
        total_bars=17520,
        train_bars=10512,
        validation_bars=3504,
        test_bars=3504,
        quality_score=0.85,
    )

    agent = SMCAnalystAgent(mock_llm)
    result = await agent.invoke(base_state)

    assert "candidate_features" in result
    features = result["candidate_features"][0]
    assert len(features.feature_names) > 0
    assert "order_block_bullish" in features.feature_names


@pytest.mark.asyncio
async def test_ml_engineer_feature_engineering(mock_llm, base_state):
    """Test ML engineer feature engineering."""
    base_state.dataset = DatasetInfo(
        symbol="EURUSD",
        timeframe="H1",
        start_date=datetime.utcnow() - timedelta(days=730),
        end_date=datetime.utcnow(),
        total_bars=17520,
        train_bars=10512,
        validation_bars=3504,
        test_bars=3504,
        quality_score=0.85,
    )

    agent = MLEngineerAgent(mock_llm)
    result = await agent.invoke(base_state)

    assert "candidate_features" in result
    features = result["candidate_features"][0]
    assert len(features.feature_names) > 0


@pytest.mark.asyncio
async def test_validation_agent(mock_llm, base_state):
    """Test validation agent model validation."""
    from trading_orchestrator.state import ModelConfig, ModelMetrics, FeatureSet

    # Setup state with trained model
    base_state.best_model = ModelConfig(
        model_type="lightgbm",
        metrics=ModelMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.12,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=200,
            overfitting_score=0.15,
        ),
        feature_set=FeatureSet(feature_names=["feature1", "feature2"]),
    )

    agent = ValidationAgent(mock_llm)
    result = await agent.invoke(base_state)

    assert "final_validation" in result
    validation = result["final_validation"]
    assert validation.robustness_score >= 0.0
    assert validation.statistical_significance >= 0.0


@pytest.mark.asyncio
async def test_risk_manager_assessment(mock_llm, base_state):
    """Test risk manager model risk assessment."""
    from trading_orchestrator.state import ModelConfig, ModelMetrics

    base_state.best_model = ModelConfig(
        model_type="lightgbm",
        metrics=ModelMetrics(
            sharpe_ratio=1.8,
            max_drawdown=0.10,
            win_rate=0.58,
            profit_factor=2.1,
            total_trades=300,
        ),
    )

    agent = RiskManagerAgent(mock_llm)
    result = await agent.invoke(base_state)

    assert "risk_assessments" in result
    assert "model" in result["risk_assessments"]

    risk = result["risk_assessments"]["model"]
    assert 0.0 <= risk.kelly_fraction <= 1.0
    assert risk.max_position_size > 0.0


@pytest.mark.asyncio
async def test_signal_evaluator(mock_llm, base_state):
    """Test signal evaluator signal generation."""
    from trading_orchestrator.state import ModelConfig, ModelMetrics, DatasetInfo

    base_state.dataset = DatasetInfo(
        symbol="EURUSD",
        timeframe="H1",
        start_date=datetime.utcnow() - timedelta(days=730),
        end_date=datetime.utcnow(),
        total_bars=17520,
        test_bars=3504,
        quality_score=0.85,
    )

    base_state.best_model = ModelConfig(
        model_type="lightgbm",
        metrics=ModelMetrics(
            sharpe_ratio=1.5,
            win_rate=0.55,
            profit_factor=1.8,
        ),
    )

    agent = SignalEvaluatorAgent(mock_llm)
    result = await agent.invoke(base_state)

    assert "generated_signals" in result
    assert len(result["generated_signals"]) > 0

    signal = result["generated_signals"][0]
    assert signal.symbol == "EURUSD"
    assert signal.direction in ["long", "short"]
    assert 0.0 <= signal.confidence <= 1.0
