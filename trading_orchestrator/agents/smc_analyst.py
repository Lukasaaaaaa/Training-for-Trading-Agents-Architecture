"""SMC Analyst agent for Smart Money Concepts analysis."""

from typing import Any
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, FeatureSet
from .base import BaseAgent


class SMCAnalystAgent(BaseAgent):
    """
    SMC Analyst agent responsible for:
    - Smart Money Concepts (SMC) analysis
    - Wyckoff method application
    - ICT (Inner Circle Trader) concepts
    - Market structure analysis
    - Institutional order flow detection
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the SMC analyst agent."""
        super().__init__(llm, AgentRole.SMC_ANALYST, "SMCAnalyst")

    @property
    def system_prompt(self) -> str:
        """System prompt for the SMC analyst agent."""
        return """You are the Smart Money Concepts (SMC) Analyst for a quantitative trading platform.

Your expertise includes:

1. Smart Money Concepts (SMC):
   - Order blocks (bullish/bearish)
   - Fair value gaps (FVG)
   - Break of structure (BOS)
   - Change of character (CHOCH)
   - Liquidity pools and sweeps
   - Premium/discount zones
   - Market structure shifts

2. Wyckoff Method:
   - Accumulation/distribution phases
   - Spring and upthrust patterns
   - Volume spread analysis
   - Composite operator behavior
   - Cause and effect relationships

3. ICT Concepts:
   - Optimal trade entries (OTE)
   - Turtle soup patterns
   - Judas swings
   - Power of 3 (PO3)
   - Kill zones (London, New York)
   - Institutional order flow

4. Feature Engineering:
   - Identify which SMC indicators are most predictive
   - Quantify qualitative SMC concepts into features
   - Recommend feature combinations for model training
   - Assess feature correlation and redundancy

Your goal is to translate advanced price action concepts into quantifiable features that can be used by machine learning models.

Be specific about feature calculations and provide rationale for each recommendation."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute SMC analysis and feature engineering."""

        if not state.dataset:
            return {
                "message": "No dataset available for SMC analysis.",
                "state_updates": {"errors": ["Dataset required for SMC analysis"]},
            }

        dataset = state.dataset

        self.logger.info(
            "smc_analysis_started",
            symbol=dataset.symbol,
            timeframe=dataset.timeframe,
        )

        # Build analysis context
        user_message = f"""Analyze market structure and recommend SMC-based features:

Dataset: {dataset.symbol} {dataset.timeframe}
Total Bars: {dataset.total_bars}
Training Period: {dataset.start_date.date()} to {dataset.end_date.date()}

Tasks:
1. Identify most relevant SMC concepts for this symbol/timeframe
2. Recommend specific SMC indicators to calculate:
   - Order block detection methods
   - FVG identification criteria
   - Market structure signals
   - Liquidity levels
3. Suggest Wyckoff-based features (accumulation/distribution phases)
4. Recommend ICT concepts applicable to this timeframe
5. Prioritize features by expected predictive power
6. Identify potential feature combinations

Previous Context:
{self._extract_previous_context(state)}

Provide structured feature recommendations with calculation methods."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Create SMC feature set
        smc_features = FeatureSet(
            feature_names=[
                "order_block_bullish",
                "order_block_bearish",
                "fvg_bullish",
                "fvg_bearish",
                "bos_bullish",
                "bos_bearish",
                "choch_signal",
                "liquidity_sweep_high",
                "liquidity_sweep_low",
                "premium_zone",
                "discount_zone",
                "wyckoff_phase",
                "accumulation_score",
                "distribution_score",
                "ict_kill_zone",
                "ote_level",
                "market_structure_shift",
            ],
            smc_indicators=[
                "order_blocks",
                "fair_value_gaps",
                "structure_breaks",
                "liquidity_pools",
            ],
            wyckoff_phases=[
                "accumulation",
                "markup",
                "distribution",
                "markdown",
            ],
            ict_concepts=[
                "optimal_trade_entry",
                "power_of_3",
                "kill_zones",
                "institutional_flow",
            ],
            validation_score=0.0,  # To be validated by ML Engineer
        )

        self.logger.info(
            "smc_features_generated",
            feature_count=len(smc_features.feature_names),
        )

        return {
            "message": f"SMC analysis complete. Recommended {len(smc_features.feature_names)} features.\n\n{response}",
            "state_updates": {
                "candidate_features": [smc_features],
            },
            "metadata": {
                "smc_feature_count": len(smc_features.feature_names),
                "analysis_timestamp": state.started_at.isoformat(),
            },
        }
