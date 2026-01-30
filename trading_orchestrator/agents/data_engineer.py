"""Data Engineer agent for data preparation and Forexsb integration."""

from typing import Any
from datetime import datetime, timedelta
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, DatasetInfo, WorkflowStage
from .base import BaseAgent


class DataEngineerAgent(BaseAgent):
    """
    Data Engineer agent responsible for:
    - Historical data acquisition and validation
    - Forexsb integration and data synchronization
    - Data quality checks and preprocessing
    - Train/validation/test split management
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the data engineer agent."""
        super().__init__(llm, AgentRole.DATA_ENGINEER, "DataEngineer")

    @property
    def system_prompt(self) -> str:
        """System prompt for the data engineer agent."""
        return """You are the Data Engineer Agent for a quantitative trading platform.

Your responsibilities:
1. Acquire and validate historical trading data
2. Integrate with Forex Strategy Builder (Forexsb) for data synchronization
3. Perform comprehensive data quality checks:
   - Missing data detection
   - Outlier identification
   - Data consistency validation
   - Sufficient data volume verification
4. Create proper train/validation/test splits (e.g., 60/20/20)
5. Prepare data pipelines for feature engineering
6. Calculate and report data quality metrics

Data Quality Standards:
- Minimum 2 years of historical data for training
- No more than 1% missing bars
- Price data within reasonable ranges (no extreme outliers)
- Consistent timeframe intervals
- Volume data validation (if available)

Output Format:
Provide structured analysis including:
- Data statistics (total bars, date ranges)
- Quality score (0-1 scale)
- Split sizes and dates
- Any data issues or warnings
- Recommendations for proceeding

Be thorough but concise in your analysis."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute data engineering tasks."""

        # Extract task parameters
        params = state.task_parameters
        symbol = params.get("symbol", "EURUSD")
        timeframe = params.get("timeframe", "H1")
        lookback_days = params.get("lookback_days", 730)  # 2 years default

        # If dataset already exists, validate it
        if state.dataset:
            return await self._validate_existing_dataset(state)

        # Otherwise, prepare new dataset
        return await self._prepare_new_dataset(state, symbol, timeframe, lookback_days)

    async def _prepare_new_dataset(
        self,
        state: AgentState,
        symbol: str,
        timeframe: str,
        lookback_days: int,
    ) -> dict[str, Any]:
        """Prepare a new dataset from scratch."""

        self.logger.info(
            "preparing_dataset",
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
        )

        # Build context for LLM
        user_message = f"""Prepare trading dataset with these parameters:

Symbol: {symbol}
Timeframe: {timeframe}
Lookback Period: {lookback_days} days
End Date: {datetime.utcnow().date()}

Tasks:
1. Calculate expected number of bars based on timeframe
2. Determine train/validation/test split dates
3. Estimate data quality score based on typical data availability
4. Identify potential data issues for this symbol/timeframe combination
5. Recommend Forexsb integration settings

Provide your analysis in structured format covering all data preparation aspects."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Create dataset info (in production, this would fetch real data)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # Calculate approximate bars based on timeframe
        bars_per_day = self._estimate_bars_per_day(timeframe)
        total_bars = int(lookback_days * bars_per_day)

        # Create train/val/test split
        train_bars = int(total_bars * 0.6)
        val_bars = int(total_bars * 0.2)
        test_bars = total_bars - train_bars - val_bars

        dataset = DatasetInfo(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_bars=total_bars,
            train_bars=train_bars,
            validation_bars=val_bars,
            test_bars=test_bars,
            quality_score=0.85,  # Placeholder
            forexsb_integration=state.task_parameters.get("use_forexsb", False),
            metadata={
                "data_source": "forexsb" if state.task_parameters.get("use_forexsb") else "broker",
                "preparation_timestamp": datetime.utcnow().isoformat(),
            },
        )

        self.logger.info(
            "dataset_prepared",
            symbol=symbol,
            total_bars=total_bars,
            quality_score=dataset.quality_score,
        )

        return {
            "message": f"Dataset prepared: {symbol} {timeframe}, {total_bars} bars, "
                      f"quality score: {dataset.quality_score:.2f}\n\n{response}",
            "state_updates": {
                "dataset": dataset,
                "stage": WorkflowStage.FEATURE_DISCOVERY,
            },
            "metadata": {
                "dataset_info": dataset.model_dump(),
            },
        }

    async def _validate_existing_dataset(self, state: AgentState) -> dict[str, Any]:
        """Validate an existing dataset."""

        dataset = state.dataset
        if not dataset:
            return {
                "message": "No dataset found to validate.",
                "state_updates": {"errors": ["No dataset available"]},
            }

        self.logger.info(
            "validating_dataset",
            symbol=dataset.symbol,
            total_bars=dataset.total_bars,
        )

        user_message = f"""Validate existing dataset:

Symbol: {dataset.symbol}
Timeframe: {dataset.timeframe}
Date Range: {dataset.start_date.date()} to {dataset.end_date.date()}
Total Bars: {dataset.total_bars}
Train/Val/Test: {dataset.train_bars}/{dataset.validation_bars}/{dataset.test_bars}
Current Quality Score: {dataset.quality_score:.2f}

Perform validation checks:
1. Is data volume sufficient for training?
2. Are splits appropriate?
3. Is quality score acceptable (>0.7)?
4. Any concerns about data freshness?
5. Should we proceed with this dataset?

Provide validation results and recommendations."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Check if validation passed
        validation_passed = dataset.quality_score >= 0.7 and dataset.total_bars >= 1000

        if validation_passed:
            return {
                "message": f"Dataset validation passed.\n\n{response}",
                "state_updates": {},
                "metadata": {"validation_passed": True},
            }
        else:
            return {
                "message": f"Dataset validation failed. Issues detected.\n\n{response}",
                "state_updates": {
                    "warnings": [
                        f"Dataset quality score {dataset.quality_score:.2f} below threshold"
                    ]
                },
                "metadata": {"validation_passed": False},
            }

    def _estimate_bars_per_day(self, timeframe: str) -> float:
        """Estimate number of bars per day based on timeframe."""
        timeframe_mapping = {
            "M1": 1440,
            "M5": 288,
            "M15": 96,
            "M30": 48,
            "H1": 24,
            "H4": 6,
            "D1": 1,
        }
        return timeframe_mapping.get(timeframe.upper(), 24)
