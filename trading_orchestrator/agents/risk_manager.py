"""Risk Manager agent for position sizing and risk assessment."""

from typing import Any
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, RiskAssessment
from .base import BaseAgent


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager agent responsible for:
    - Position sizing calculations
    - Portfolio mathematics
    - Kelly criterion application
    - Value at Risk (VaR) and CVaR
    - Correlation analysis
    - Risk-adjusted returns
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the risk manager agent."""
        super().__init__(llm, AgentRole.RISK_MANAGER, "RiskManager")

    @property
    def system_prompt(self) -> str:
        """System prompt for the risk manager agent."""
        return """You are the Risk Manager Agent for a quantitative trading platform.

Your expertise includes:

1. Position Sizing:
   - Fixed fractional position sizing
   - Kelly criterion and fractional Kelly
   - Volatility-based position sizing
   - Risk parity approaches
   - Maximum position limits

2. Portfolio Mathematics:
   - Portfolio variance and covariance
   - Correlation analysis
   - Diversification benefits
   - Portfolio optimization
   - Rebalancing strategies

3. Risk Metrics:
   - Value at Risk (VaR) - historical and parametric
   - Conditional Value at Risk (CVaR)
   - Maximum drawdown analysis
   - Sharpe and Sortino ratios
   - Risk-adjusted performance

4. Risk Management Rules:
   - Maximum risk per trade (typically 1-2%)
   - Maximum portfolio risk (typically 6-10%)
   - Correlation limits between positions
   - Exposure limits per asset/sector
   - Drawdown-based risk reduction

5. Signal Risk Assessment:
   - Confidence-based sizing
   - Stop loss placement validation
   - Risk-reward ratio evaluation
   - Margin requirement calculations

Your goal is to ensure sustainable risk-adjusted returns while protecting capital.

Be specific with calculations and provide clear risk limits."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute risk management tasks."""

        # Check if we have signals to assess
        if state.filtered_signals:
            return await self._assess_signal_risk(state)
        elif state.best_model:
            return await self._assess_model_risk(state)
        else:
            return {
                "message": "No signals or models available for risk assessment.",
                "state_updates": {"warnings": ["Risk assessment requires signals or model"]},
            }

    async def _assess_model_risk(self, state: AgentState) -> dict[str, Any]:
        """Assess risk characteristics of the trained model."""

        self.logger.info("model_risk_assessment_started")

        model = state.best_model
        assert model is not None

        metrics = model.metrics

        user_message = f"""Assess risk profile of trained trading model:

Model Performance:
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Maximum Drawdown: {metrics.max_drawdown:.2%}
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Total Trades: {metrics.total_trades}
- Overfitting Score: {metrics.overfitting_score:.2f}

Tasks:
1. Evaluate risk-adjusted performance:
   - Is Sharpe ratio acceptable (>1.5 preferred)?
   - Is maximum drawdown within limits (<15% preferred)?
   - Is profit factor sustainable (>1.5 preferred)?

2. Recommend position sizing approach:
   - Fixed fractional size
   - Kelly criterion fraction
   - Volatility-adjusted sizing

3. Calculate risk parameters:
   - Maximum position size (% of equity)
   - Stop loss requirements
   - Leverage limits

4. Identify risk concerns:
   - Overfitting indicators
   - Concentration risk
   - Market regime dependencies

Provide comprehensive risk assessment and recommendations."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Calculate Kelly fraction (simplified)
        win_rate = metrics.win_rate
        profit_factor = metrics.profit_factor
        avg_win = profit_factor * (1 - win_rate) if win_rate < 1 else profit_factor
        avg_loss = 1.0

        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Volatility-based sizing
        max_position = 0.02  # 2% risk per trade (conservative)

        # Create risk assessment
        risk_assessment = RiskAssessment(
            max_position_size=max_position,
            kelly_fraction=kelly_fraction,
            var_95=metrics.max_drawdown * 0.5,  # Simplified VaR estimate
            cvar_95=metrics.max_drawdown * 0.7,  # Simplified CVaR estimate
            risk_score=self._calculate_risk_score(metrics),
            approved=metrics.sharpe_ratio > 1.0 and metrics.max_drawdown < 0.20,
        )

        if not risk_assessment.approved:
            risk_assessment.rejection_reason = (
                f"Risk metrics do not meet thresholds: "
                f"Sharpe {metrics.sharpe_ratio:.2f} (min 1.0), "
                f"MaxDD {metrics.max_drawdown:.2%} (max 20%)"
            )

        self.logger.info(
            "model_risk_assessed",
            approved=risk_assessment.approved,
            kelly_fraction=kelly_fraction,
            max_position_size=max_position,
        )

        return {
            "message": f"Risk assessment complete. "
                      f"{'Approved' if risk_assessment.approved else 'Rejected'}.\n"
                      f"Kelly Fraction: {kelly_fraction:.2%}, "
                      f"Max Position: {max_position:.2%}\n\n{response}",
            "state_updates": {
                "risk_assessments": {"model": risk_assessment},
            },
            "metadata": {
                "risk_metrics": risk_assessment.model_dump(),
            },
        }

    async def _assess_signal_risk(self, state: AgentState) -> dict[str, Any]:
        """Assess risk for individual trading signals."""

        self.logger.info(
            "signal_risk_assessment_started",
            signal_count=len(state.filtered_signals),
        )

        signals = state.filtered_signals
        model = state.best_model

        user_message = f"""Assess risk for trading signals:

Number of Signals: {len(signals)}
Model Sharpe: {model.metrics.sharpe_ratio:.2f if model else 'N/A'}
Model Max DD: {model.metrics.max_drawdown:.2% if model else 'N/A'}

Signal Summary:
{self._format_signal_summary(signals[:5])}  # First 5 signals

Tasks:
1. Validate risk-reward ratios for each signal
2. Calculate appropriate position sizes
3. Assess portfolio correlation if multiple signals
4. Identify any concentration risks
5. Recommend signal prioritization based on risk-adjusted potential

Provide risk assessment for signal execution."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Assess each signal
        risk_assessments = {}
        for signal in signals:
            assessment = RiskAssessment(
                max_position_size=0.02,  # 2% per trade
                kelly_fraction=0.05,  # Conservative 5%
                risk_score=self._calculate_signal_risk_score(signal),
                approved=signal.confidence > 0.6 and signal.risk_reward_ratio >= 2.0,
            )

            if not assessment.approved:
                assessment.rejection_reason = (
                    f"Signal does not meet criteria: "
                    f"Confidence {signal.confidence:.2%} (min 60%), "
                    f"R:R {signal.risk_reward_ratio:.2f} (min 2.0)"
                )

            risk_assessments[signal.signal_id] = assessment

        approved_count = sum(1 for a in risk_assessments.values() if a.approved)

        self.logger.info(
            "signal_risk_assessed",
            total_signals=len(signals),
            approved_signals=approved_count,
        )

        return {
            "message": f"Signal risk assessment complete. "
                      f"{approved_count}/{len(signals)} signals approved.\n\n{response}",
            "state_updates": {
                "risk_assessments": risk_assessments,
            },
            "metadata": {
                "approved_signals": approved_count,
                "total_signals": len(signals),
            },
        }

    def _calculate_risk_score(self, metrics: Any) -> float:
        """Calculate overall risk score (0-1, higher is riskier)."""
        # Combine multiple risk factors
        drawdown_score = min(metrics.max_drawdown / 0.30, 1.0)  # Normalize to 30% max
        sharpe_score = max(0.0, 1.0 - metrics.sharpe_ratio / 3.0)  # Higher Sharpe = lower risk
        overfitting_score = metrics.overfitting_score

        risk_score = (drawdown_score * 0.4 + sharpe_score * 0.3 + overfitting_score * 0.3)
        return min(1.0, max(0.0, risk_score))

    def _calculate_signal_risk_score(self, signal: Any) -> float:
        """Calculate risk score for individual signal."""
        confidence_risk = 1.0 - signal.confidence
        rr_risk = max(0.0, 1.0 - signal.risk_reward_ratio / 3.0)

        return (confidence_risk * 0.6 + rr_risk * 0.4)

    def _format_signal_summary(self, signals: list[Any]) -> str:
        """Format signals for display."""
        if not signals:
            return "No signals to display"

        summary_lines = []
        for signal in signals:
            summary_lines.append(
                f"- {signal.direction.upper()}: Confidence {signal.confidence:.2%}, "
                f"R:R {signal.risk_reward_ratio:.2f}"
            )

        return "\n".join(summary_lines)
