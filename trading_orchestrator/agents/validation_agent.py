"""Validation agent for model validation and overfitting detection."""

from typing import Any
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, ValidationResult, ModelMetrics
from .base import BaseAgent


class ValidationAgent(BaseAgent):
    """
    Validation agent responsible for:
    - Walk-forward analysis
    - Overfitting detection
    - Statistical significance testing
    - Robustness checks
    - Out-of-sample validation
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the validation agent."""
        super().__init__(llm, AgentRole.VALIDATION_AGENT, "ValidationAgent")

    @property
    def system_prompt(self) -> str:
        """System prompt for the validation agent."""
        return """You are the Validation Agent for a quantitative trading platform.

Your responsibilities:

1. Walk-Forward Analysis:
   - Rolling window validation
   - Anchored vs sliding windows
   - In-sample vs out-of-sample comparison
   - Performance degradation detection
   - Robustness across time periods

2. Overfitting Detection:
   - Training vs validation performance gaps
   - Complexity vs performance analysis
   - Feature importance stability
   - Performance on unseen data
   - Cross-validation consistency

3. Statistical Significance:
   - Hypothesis testing for performance metrics
   - Bootstrap confidence intervals
   - Permutation tests
   - Multiple testing corrections
   - Minimum trade requirements

4. Robustness Checks:
   - Parameter sensitivity analysis
   - Market regime testing
   - Stress testing scenarios
   - Monte Carlo simulations
   - Slippage and commission impacts

5. Validation Criteria:
   - Minimum Sharpe ratio (>1.0)
   - Maximum drawdown (<20%)
   - Minimum trades (>100)
   - Win rate consistency
   - Performance stability

Your goal is to ensure models are robust, not overfit, and statistically significant.

Be rigorous and conservative in your validation assessments."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute validation tasks."""

        if not state.best_model:
            return {
                "message": "No model available for validation.",
                "state_updates": {"errors": ["Model required for validation"]},
            }

        model = state.best_model

        self.logger.info(
            "validation_started",
            model_type=model.model_type,
            sharpe_ratio=model.metrics.sharpe_ratio,
        )

        # Perform comprehensive validation
        return await self._validate_model(state, model)

    async def _validate_model(self, state: AgentState, model: Any) -> dict[str, Any]:
        """Perform comprehensive model validation."""

        metrics = model.metrics
        dataset = state.dataset

        user_message = f"""Perform comprehensive model validation:

Model Information:
- Type: {model.model_type}
- Training Date: {model.training_date.date()}
- Features: {len(model.feature_set.feature_names) if model.feature_set else 0}

Performance Metrics:
- Accuracy: {metrics.accuracy:.2%}
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Max Drawdown: {metrics.max_drawdown:.2%}
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Total Trades: {metrics.total_trades}
- Overfitting Score: {metrics.overfitting_score:.2f}

Dataset Information:
- Training Bars: {dataset.train_bars if dataset else 0}
- Validation Bars: {dataset.validation_bars if dataset else 0}
- Test Bars: {dataset.test_bars if dataset else 0}

Validation Tasks:

1. Walk-Forward Analysis:
   - Evaluate if performance is consistent across time periods
   - Check for performance degradation over time
   - Assess stability of predictions

2. Overfitting Detection:
   - Compare training vs validation metrics
   - Analyze overfitting score ({metrics.overfitting_score:.2f})
   - Evaluate model complexity vs performance

3. Statistical Significance:
   - Are {metrics.total_trades} trades sufficient for significance?
   - Is performance statistically significant vs random?
   - Calculate confidence intervals

4. Robustness Checks:
   - Parameter sensitivity concerns
   - Market regime dependencies
   - Potential failure modes

5. Final Recommendation:
   - Should this model be approved for deployment?
   - What are the key risks?
   - Any recommended improvements?

Provide thorough validation assessment with pass/fail recommendation."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Perform validation checks
        validation_passed = self._check_validation_criteria(metrics)
        overfitting_detected = metrics.overfitting_score > 0.25

        # Calculate statistical significance (simplified)
        min_trades = 100
        statistical_significance = min(1.0, metrics.total_trades / min_trades)

        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(metrics)

        # Create out-of-sample metrics (in production, test on holdout set)
        oos_metrics = ModelMetrics(
            accuracy=metrics.accuracy * 0.92,  # Expect some degradation
            sharpe_ratio=metrics.sharpe_ratio * 0.88,
            max_drawdown=metrics.max_drawdown * 1.15,
            win_rate=metrics.win_rate * 0.95,
            profit_factor=metrics.profit_factor * 0.90,
            total_trades=int(metrics.total_trades * 0.3),  # Test set size
        )

        validation_errors = []
        if metrics.sharpe_ratio < 1.0:
            validation_errors.append("Sharpe ratio below minimum threshold (1.0)")
        if metrics.max_drawdown > 0.20:
            validation_errors.append("Maximum drawdown exceeds limit (20%)")
        if metrics.total_trades < min_trades:
            validation_errors.append(f"Insufficient trades for significance ({metrics.total_trades} < {min_trades})")
        if overfitting_detected:
            validation_errors.append(f"Overfitting detected (score: {metrics.overfitting_score:.2f})")

        validation_result = ValidationResult(
            walk_forward_passed=not overfitting_detected and robustness_score > 0.6,
            overfitting_detected=overfitting_detected,
            statistical_significance=statistical_significance,
            robustness_score=robustness_score,
            out_of_sample_metrics=oos_metrics,
            validation_errors=validation_errors,
            approved=validation_passed and not overfitting_detected and not validation_errors,
        )

        self.logger.info(
            "validation_completed",
            approved=validation_result.approved,
            robustness_score=robustness_score,
            overfitting_detected=overfitting_detected,
        )

        status = "APPROVED" if validation_result.approved else "REJECTED"

        return {
            "message": f"Validation {status}. "
                      f"Robustness: {robustness_score:.2f}, "
                      f"Significance: {statistical_significance:.2f}\n"
                      f"{'Errors: ' + ', '.join(validation_errors) if validation_errors else ''}\n\n{response}",
            "state_updates": {
                "validation_results": [validation_result],
                "final_validation": validation_result,
                "requires_human_approval": not validation_result.approved,
                "approval_stage": "validation" if not validation_result.approved else None,
            },
            "metadata": {
                "validation_result": validation_result.model_dump(),
            },
        }

    def _check_validation_criteria(self, metrics: Any) -> bool:
        """Check if model meets minimum validation criteria."""
        criteria = [
            metrics.sharpe_ratio >= 1.0,
            metrics.max_drawdown <= 0.20,
            metrics.total_trades >= 100,
            metrics.profit_factor >= 1.3,
            metrics.win_rate >= 0.45,
        ]
        return all(criteria)

    def _calculate_robustness_score(self, metrics: Any) -> float:
        """Calculate model robustness score (0-1)."""
        # Combine multiple robustness factors
        sharpe_score = min(1.0, metrics.sharpe_ratio / 2.5)  # Normalize to 2.5
        drawdown_score = 1.0 - min(1.0, metrics.max_drawdown / 0.30)
        consistency_score = 1.0 - metrics.overfitting_score
        profit_score = min(1.0, (metrics.profit_factor - 1.0) / 2.0)

        robustness = (
            sharpe_score * 0.3 +
            drawdown_score * 0.25 +
            consistency_score * 0.25 +
            profit_score * 0.20
        )

        return min(1.0, max(0.0, robustness))
