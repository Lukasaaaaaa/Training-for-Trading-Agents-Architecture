# Trading Bot Multi-Agent System

## Project Overview
A comprehensive multi-agent architecture for algorithmic trading, combining market microstructure analysis, Smart Money Concepts (SMC), and machine learning (LightGBM) for signal generation and risk management.

## Architecture Philosophy
- **Modular Design:** Each agent has single, focused responsibility
- **Context Isolation:** Agents gather their own context when activated
- **File-Based Communication:** Agents share state through structured files
- **Consensus Mechanisms:** Trading decisions require multi-agent agreement

## Available Subagents

### Domain Specialists
| Agent | Expertise | When to Invoke |
|-------|-----------|----------------|
| `market-microstructure-analyst` | Order flow, liquidity, depth | Analyzing institutional activity, volume patterns, bid-ask dynamics |
| `smc-pattern-recognizer` | Smart Money Concepts, ICT | Order blocks, FVGs, market structure, liquidity sweeps |
| `ml-model-engineer` | LightGBM, ML development | Model training, hyperparameter tuning, predictions |
| `feature-engineer` | Feature creation/selection | Creating features, transformations, feature selection |
| `risk-manager` | Position sizing, drawdown | MANDATORY before any trade execution |
| `backtester` | Validation, walk-forward | Strategy validation before deployment |
| `signal-evaluator` | Signal aggregation, quality | Combining signals, conflict resolution, final scoring |

### Coordination Agents
| Agent | Role | When to Invoke |
|-------|------|----------------|
| `trading-coordinator` | Master orchestrator | Multi-agent workflows, complex decisions |
| `data-pipeline-manager` | Data infrastructure | Data ingestion, quality issues, preprocessing |
| `model-trainer` | Training orchestration | Training cycles, retraining triggers, model lifecycle |

### Utility Agents
| Agent | Role | When to Invoke |
|-------|------|----------------|
| `discussion-facilitator` | Multi-agent discussions | Facilitating structured discussions between agents |

## Quick Reference: Agent Locations
All agent definitions are in `./agents/`:
- `market-microstructure-analyst.md`
- `smc-pattern-recognizer.md`
- `ml-model-engineer.md`
- `feature-engineer.md`
- `risk-manager.md`
- `backtester.md`
- `signal-evaluator.md`
- `trading-coordinator.md`
- `data-pipeline-manager.md`
- `model-trainer.md`
- `discussion-facilitator.md`

## Configuration Files
All configs in `./config/`:
- `risk_config.yaml` - Risk limits, drawdown scaling, circuit breakers
- `ml_config.yaml` - LightGBM params, training settings
- `smc_config.yaml` - SMC pattern detection settings
- `microstructure_config.yaml` - Order flow analysis settings
- `feature_config.yaml` - Feature engineering settings
- `backtest_config.yaml` - Validation and WFO settings
- `signal_config.yaml` - Signal aggregation settings
- `evaluation_framework.yaml` - Agent evaluation and scoring framework

## Communication Protocol
Agents communicate through structured state files in `/state/`:
- `/state/signals/` - Trading signals from various sources
- `/state/features/` - Engineered features for ML models
- `/state/risk/` - Risk assessments and limits
- `/state/consensus/` - Multi-agent voting records
- `/state/models/` - Model artifacts and metadata
- `/state/training/` - Training status and runs
- `/state/data/` - Data status and market context

## Consensus Building
Trading decisions require agreement from:
1. At least one pattern agent (SMC or microstructure)
2. ML signal evaluation
3. Risk manager approval (MANDATORY)

## Key Workflows

### Real-Time Signal Generation
```
Data Pipeline -> [Microstructure | SMC | Features] -> ML Model ->
Signal Evaluator -> Risk Manager -> Execute/Notify
```

### Model Training Cycle
```
Data Prep -> Feature Engineering -> ML Training -> Backtester Validation ->
Risk Assessment -> Deployment Decision
```

## Directory Structure
```
/agents/              - Agent definitions (11 agents)
/config/              - Configuration files (8 configs)
/state/               - Runtime state and communication
/data/                - Market data storage
/models/              - Trained model artifacts
/backtests/           - Backtest results and reports
/logs/                - Execution logs
/trading_orchestrator/ - Python implementation (LangGraph)
/examples/            - Usage examples
/tests/               - Test suite
/knowledge/           - Academic knowledge base
/monitoring/          - Prometheus & Grafana configs
```

## Activation Guidelines
- Delegate to domain specialists for their expertise areas
- ALWAYS involve risk-manager before trade execution
- Use trading-coordinator for multi-agent workflows
- Run backtester before deploying any strategy changes
- Check data-pipeline-manager for data quality issues

## Signal Quality Tiers
| Tier | Score | Action |
|------|-------|--------|
| A | 0.80-1.00 | High conviction trade |
| B | 0.65-0.80 | Normal trade |
| C | 0.50-0.65 | Reduced size |
| D | 0.35-0.50 | Monitor only |
| F | 0.00-0.35 | No trade |

## Risk Limits (Default)
- Max risk per trade: 0.5%
- Max daily drawdown: 3%
- Max total drawdown: 20% (trading halted)
- Minimum risk-reward: 1.5:1
