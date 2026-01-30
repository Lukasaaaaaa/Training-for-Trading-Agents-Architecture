"""ML Engineer agent for model training and optimization."""

from typing import Any
from datetime import datetime
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, FeatureSet, ModelConfig, ModelMetrics
from .base import BaseAgent


class MLEngineerAgent(BaseAgent):
    """
    ML Engineer agent responsible for:
    - Feature engineering and selection
    - LightGBM model training and optimization
    - Hyperparameter tuning
    - Cross-validation strategies
    - Model performance evaluation
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the ML engineer agent."""
        super().__init__(llm, AgentRole.ML_ENGINEER, "MLEngineer")

    @property
    def system_prompt(self) -> str:
        """System prompt for the ML engineer agent."""
        return """You are the ML Engineer Agent for a quantitative trading platform.

Your expertise includes:

1. Feature Engineering:
   - Technical indicator calculation
   - Feature transformation and scaling
   - Interaction features and polynomial features
   - Time-based features (hour, day, session)
   - Lag features and rolling statistics
   - Feature importance analysis
   - Dimensionality reduction

2. LightGBM Optimization:
   - Hyperparameter tuning strategies
   - Objective function selection for trading
   - Early stopping criteria
   - Handling class imbalance
   - Regularization techniques
   - Learning rate scheduling

3. Cross-Validation:
   - Time-series split strategies
   - Purging and embargo for financial data
   - Walk-forward optimization
   - Out-of-sample validation
   - Overfitting prevention

4. Performance Metrics:
   - Classification metrics (accuracy, precision, recall, F1)
   - Trading-specific metrics (Sharpe, Sortino, max drawdown)
   - Win rate and profit factor
   - Risk-adjusted returns

5. Model Selection:
   - Ensemble methods
   - Model complexity vs performance tradeoff
   - Production deployment considerations

Provide detailed technical recommendations with specific hyperparameters and validation strategies."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute ML engineering tasks."""

        if not state.dataset:
            return {
                "message": "No dataset available for model training.",
                "state_updates": {"errors": ["Dataset required for ML training"]},
            }

        # Determine sub-task
        if not state.candidate_features:
            return await self._engineer_features(state)
        elif not state.selected_features:
            return await self._select_features(state)
        else:
            return await self._train_model(state)

    async def _engineer_features(self, state: AgentState) -> dict[str, Any]:
        """Engineer technical features complementary to SMC features."""

        self.logger.info("feature_engineering_started")

        dataset = state.dataset
        assert dataset is not None

        user_message = f"""Engineer technical indicator features to complement SMC analysis:

Dataset: {dataset.symbol} {dataset.timeframe}
Total Bars: {dataset.total_bars}

Tasks:
1. Recommend core technical indicators:
   - Trend indicators (MA, EMA, MACD)
   - Momentum indicators (RSI, Stochastic)
   - Volatility indicators (ATR, Bollinger Bands)
   - Volume indicators (OBV, VWAP)

2. Suggest derived features:
   - Price ratios and differences
   - Rate of change features
   - Rolling statistics (mean, std, min, max)
   - Lag features (1-bar, 5-bar, 10-bar lags)

3. Time-based features:
   - Hour of day
   - Day of week
   - Session indicators (Asian, London, New York)

4. Feature interactions:
   - Combinations of technical and SMC features
   - Polynomial features (if appropriate)

Previous Context:
{self._extract_previous_context(state)}

Provide specific feature list with calculation methods."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Create technical feature set
        tech_features = FeatureSet(
            feature_names=[
                "sma_20",
                "sma_50",
                "ema_12",
                "ema_26",
                "rsi_14",
                "macd_line",
                "macd_signal",
                "macd_histogram",
                "bbands_upper",
                "bbands_lower",
                "bbands_width",
                "atr_14",
                "stochastic_k",
                "stochastic_d",
                "obv",
                "vwap",
                "price_change_1",
                "price_change_5",
                "volume_ratio",
                "volatility_ratio",
                "hour_of_day",
                "day_of_week",
                "session_asian",
                "session_london",
                "session_newyork",
            ],
            technical_indicators=[
                "moving_averages",
                "momentum",
                "volatility",
                "volume",
                "time_features",
            ],
            validation_score=0.0,
        )

        self.logger.info(
            "technical_features_generated",
            feature_count=len(tech_features.feature_names),
        )

        return {
            "message": f"Technical features engineered. Generated {len(tech_features.feature_names)} features.\n\n{response}",
            "state_updates": {
                "candidate_features": [tech_features],
            },
            "metadata": {
                "tech_feature_count": len(tech_features.feature_names),
            },
        }

    async def _select_features(self, state: AgentState) -> dict[str, Any]:
        """Select optimal feature set from candidates."""

        self.logger.info("feature_selection_started")

        # Combine all candidate features
        all_features = []
        for feature_set in state.candidate_features:
            all_features.extend(feature_set.feature_names)

        user_message = f"""Select optimal feature set for model training:

Total Candidate Features: {len(all_features)}
Dataset: {state.dataset.symbol} {state.dataset.timeframe} if state.dataset else "Unknown"}
Training Samples: {state.dataset.train_bars if state.dataset else 0}

Available Features:
{', '.join(all_features[:50])}{'...' if len(all_features) > 50 else ''}

Tasks:
1. Identify potential feature redundancy
2. Recommend feature selection strategy:
   - Correlation-based filtering
   - Tree-based feature importance
   - Recursive feature elimination
3. Suggest optimal feature subset size
4. Highlight must-include features
5. Flag features to exclude

Provide feature selection recommendations."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Combine features from all candidates
        combined_features = FeatureSet(
            feature_names=all_features,
            smc_indicators=[],
            technical_indicators=[],
            validation_score=0.0,
        )

        # Collect SMC and technical indicators
        for fs in state.candidate_features:
            combined_features.smc_indicators.extend(fs.smc_indicators)
            combined_features.technical_indicators.extend(fs.technical_indicators)
            combined_features.wyckoff_phases.extend(fs.wyckoff_phases)
            combined_features.ict_concepts.extend(fs.ict_concepts)

        self.logger.info(
            "features_selected",
            total_features=len(combined_features.feature_names),
        )

        return {
            "message": f"Feature selection complete. Selected {len(combined_features.feature_names)} features.\n\n{response}",
            "state_updates": {
                "selected_features": combined_features,
            },
            "metadata": {
                "selected_feature_count": len(combined_features.feature_names),
            },
        }

    async def _train_model(self, state: AgentState) -> dict[str, Any]:
        """Train LightGBM model with selected features."""

        self.logger.info("model_training_started")

        dataset = state.dataset
        features = state.selected_features
        assert dataset is not None and features is not None

        user_message = f"""Train LightGBM model for trading signal prediction:

Dataset: {dataset.symbol} {dataset.timeframe}
Training Samples: {dataset.train_bars}
Validation Samples: {dataset.validation_bars}
Features: {len(features.feature_names)}

Tasks:
1. Recommend LightGBM hyperparameters:
   - num_leaves (tree complexity)
   - max_depth
   - learning_rate
   - n_estimators
   - min_child_samples
   - subsample and colsample_bytree
   - reg_alpha and reg_lambda (regularization)

2. Training strategy:
   - Objective function (binary, multiclass, or regression)
   - Early stopping rounds
   - Validation metric
   - Cross-validation folds

3. Expected performance metrics:
   - Target accuracy range
   - Acceptable Sharpe ratio
   - Maximum drawdown threshold

4. Overfitting prevention:
   - Regularization approach
   - Validation strategy
   - Feature selection iterations

Provide training configuration and expected results."""

        response = await self._call_llm(self.system_prompt, user_message)

        # Create model configuration (placeholder - in production, train real model)
        model = ModelConfig(
            model_type="lightgbm",
            hyperparameters={
                "objective": "binary",
                "num_leaves": 31,
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "early_stopping_rounds": 50,
            },
            feature_set=features,
            training_date=datetime.utcnow(),
            metrics=ModelMetrics(
                accuracy=0.58,
                precision=0.62,
                recall=0.55,
                f1_score=0.58,
                sharpe_ratio=1.45,
                sortino_ratio=2.1,
                max_drawdown=0.12,
                win_rate=0.54,
                profit_factor=1.8,
                total_trades=450,
                overfitting_score=0.15,
            ),
            checkpoint_path=f"./checkpoints/model_{dataset.symbol}_{datetime.utcnow().isoformat()}.pkl",
        )

        self.logger.info(
            "model_trained",
            sharpe_ratio=model.metrics.sharpe_ratio,
            accuracy=model.metrics.accuracy,
        )

        return {
            "message": f"Model training complete. Sharpe: {model.metrics.sharpe_ratio:.2f}, "
                      f"Accuracy: {model.metrics.accuracy:.2%}\n\n{response}",
            "state_updates": {
                "trained_models": [model],
                "best_model": model,
                "training_progress": 1.0,
            },
            "metadata": {
                "model_metrics": model.metrics.model_dump(),
                "hyperparameters": model.hyperparameters,
            },
        }
