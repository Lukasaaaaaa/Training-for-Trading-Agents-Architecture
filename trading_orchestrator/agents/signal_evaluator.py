"""Signal Evaluator agent for trading signal assessment and filtering."""

from typing import Any, Literal
from datetime import datetime
import uuid
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, TradingSignal
from .base import BaseAgent


class SignalEvaluatorAgent(BaseAgent):
    """
    Signal Evaluator agent responsible for:
    - Signal generation from model predictions
    - Signal quality assessment
    - Multi-timeframe confirmation
    - Filter application
    - Confidence scoring
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the signal evaluator agent."""
        super().__init__(llm, AgentRole.SIGNAL_EVALUATOR, "SignalEvaluator")

    @property
    def system_prompt(self) -> str:
        """System prompt for the signal evaluator agent."""
        return """You are the Signal Evaluator Agent for a quantitative trading platform.

Your responsibilities:

1. Signal Generation:
   - Convert model predictions into actionable signals
   - Determine entry, stop loss, and take profit levels
   - Calculate position sizing recommendations
   - Assign confidence scores

2. Signal Quality Assessment:
   - Evaluate signal strength and clarity
   - Check multi-timeframe alignment
   - Verify SMC concept alignment
   - Assess risk-reward ratios
   - Calculate expected value

3. Signal Filtering:
   - Apply quality filters (min confidence threshold)
   - Risk-reward minimum requirements
   - Market condition filters
   - Time-of-day filters (avoid news events, low liquidity)
   - Correlation filters (avoid over-concentration)

4. Confirmation Checks:
   - Trend alignment
   - Volume confirmation
   - SMC structure alignment
   - Support/resistance levels
   - Market regime suitability

5. Signal Prioritization:
   - Rank signals by expected value
   - Consider portfolio impact
   - Account for correlation
   - Optimize for diversification

Your goal is to ensure only high-quality, high-probability signals are recommended for execution.

Be selective and conservative in signal approval."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute signal evaluation tasks."""

        if not state.best_model:
            return {
                "message": "No model available for signal generation.",
                "state_updates": {"errors": ["Model required for signal generation"]},
            }

        if state.generated_signals:
            # Already have signals, filter them
            return await self._filter_signals(state)
        else:
            # Generate new signals
            return await self._generate_signals(state)

    async def _generate_signals(self, state: AgentState) -> dict[str, Any]:
        """Generate trading signals from model."""

        self.logger.info("signal_generation_started")

        model = state.best_model
        dataset = state.dataset
        assert model is not None and dataset is not None

        user_message = f"""Generate trading signals from trained model:

Model Performance:
- Sharpe Ratio: {model.metrics.sharpe_ratio:.2f}
- Win Rate: {model.metrics.win_rate:.2%}
- Profit Factor: {model.metrics.profit_factor:.2f}

Dataset:
- Symbol: {dataset.symbol}
- Timeframe: {dataset.timeframe}
- Test Bars: {dataset.test_bars}

Features:
- SMC Indicators: {', '.join(model.feature_set.smc_indicators[:5]) if model.feature_set else 'N/A'}
- Technical Indicators: Available

Tasks:
1. Simulate signal generation on test data
2. For each signal, determine:
   - Direction (long/short)
   - Entry price and timing
   - Stop loss level (based on SMC/technical)
   - Take profit levels (multiple targets)
   - Confidence score (0-1)
   - Risk-reward ratio

3. Consider signal quality factors:
   - Model confidence
   - SMC alignment
   - Technical confirmation
   - Market structure

4. Generate realistic signal set with diversity:
   - Mix of long/short signals
   - Various confidence levels
   - Different market conditions

Provide signal generation summary with key statistics."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Generate sample signals (in production, use real model predictions)
        signals = []
        num_signals = 20  # Generate 20 test signals

        for i in range(num_signals):
            direction: Literal["long", "short"] = "long" if i % 2 == 0 else "short"
            base_confidence = 0.45 + (i % 5) * 0.1  # Range: 0.45 to 0.85

            signal = TradingSignal(
                signal_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                symbol=dataset.symbol,
                direction=direction,
                confidence=base_confidence,
                entry_price=1.1000 + (i * 0.0010),  # Simulated prices
                stop_loss=1.0950 if direction == "long" else 1.1050,
                take_profit=1.1100 if direction == "long" else 1.0900,
                risk_reward_ratio=2.0 + (i % 3) * 0.5,
                smc_alignment=(i % 3) == 0,  # Every 3rd signal has SMC alignment
                filters_passed=[],
                metadata={
                    "model_version": model.model_version,
                    "generation_timestamp": datetime.utcnow().isoformat(),
                },
            )

            signals.append(signal)

        self.logger.info(
            "signals_generated",
            signal_count=len(signals),
            avg_confidence=sum(s.confidence for s in signals) / len(signals),
        )

        return {
            "message": f"Generated {len(signals)} trading signals. "
                      f"Avg confidence: {sum(s.confidence for s in signals) / len(signals):.2%}\n\n{response}",
            "state_updates": {
                "generated_signals": signals,
            },
            "metadata": {
                "signal_count": len(signals),
                "long_signals": sum(1 for s in signals if s.direction == "long"),
                "short_signals": sum(1 for s in signals if s.direction == "short"),
            },
        }

    async def _filter_signals(self, state: AgentState) -> dict[str, Any]:
        """Filter and evaluate generated signals."""

        self.logger.info(
            "signal_filtering_started",
            signal_count=len(state.generated_signals),
        )

        signals = state.generated_signals

        user_message = f"""Filter and evaluate trading signals:

Total Signals: {len(signals)}
Model Sharpe: {state.best_model.metrics.sharpe_ratio:.2f if state.best_model else 'N/A'}

Signal Summary:
{self._format_signal_statistics(signals)}

Filtering Criteria:
1. Minimum confidence threshold: 0.60 (60%)
2. Minimum risk-reward ratio: 2.0
3. SMC alignment preferred
4. Avoid over-concentration in single direction

Tasks:
1. Apply quality filters to signals
2. Rank remaining signals by quality
3. Check for portfolio correlation
4. Identify top signals for execution
5. Flag any concerns or warnings

Provide filtering results and top signal recommendations."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Apply filters
        filtered = []
        for signal in signals:
            passed_filters = []

            # Confidence filter
            if signal.confidence >= 0.60:
                passed_filters.append("confidence_threshold")

            # Risk-reward filter
            if signal.risk_reward_ratio >= 2.0:
                passed_filters.append("risk_reward_ratio")

            # SMC alignment bonus
            if signal.smc_alignment:
                passed_filters.append("smc_alignment")
                signal.confidence *= 1.1  # Boost confidence

            # Only include if passed minimum filters
            if len(passed_filters) >= 2:
                signal.filters_passed = passed_filters
                filtered.append(signal)

        # Sort by confidence
        filtered.sort(key=lambda s: s.confidence, reverse=True)

        self.logger.info(
            "signals_filtered",
            original_count=len(signals),
            filtered_count=len(filtered),
            pass_rate=len(filtered) / len(signals) if signals else 0,
        )

        return {
            "message": f"Signal filtering complete. "
                      f"{len(filtered)}/{len(signals)} signals passed filters "
                      f"({len(filtered)/len(signals):.1%} pass rate).\n\n{response}",
            "state_updates": {
                "filtered_signals": filtered,
            },
            "metadata": {
                "original_signals": len(signals),
                "filtered_signals": len(filtered),
                "pass_rate": len(filtered) / len(signals) if signals else 0,
                "top_signal_confidence": filtered[0].confidence if filtered else 0,
            },
        }

    def _format_signal_statistics(self, signals: list[TradingSignal]) -> str:
        """Format signal statistics for display."""
        if not signals:
            return "No signals available"

        long_count = sum(1 for s in signals if s.direction == "long")
        short_count = sum(1 for s in signals if s.direction == "short")
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        avg_rr = sum(s.risk_reward_ratio for s in signals) / len(signals)
        smc_aligned = sum(1 for s in signals if s.smc_alignment)

        return f"""- Long signals: {long_count}
- Short signals: {short_count}
- Average confidence: {avg_confidence:.2%}
- Average R:R: {avg_rr:.2f}
- SMC aligned: {smc_aligned} ({smc_aligned/len(signals):.1%})"""
