# Quick Reference Guide

Fast reference for common tasks and patterns in the Trading Orchestrator system.

## Installation & Setup

```bash
# Quick setup
./quickstart.sh

# Manual setup
poetry install
cp .env.example .env
# Edit .env with your API keys
```

## Basic Usage

### Run a Simple Workflow

```python
from trading_orchestrator.llm_factory import create_llm
from trading_orchestrator.graph import TradingOrchestrator

llm = create_llm()
orchestrator = TradingOrchestrator(llm)
compiled = await orchestrator.compile()

state = orchestrator.create_initial_state(
    task_description="Develop EURUSD H1 trading bot",
    task_parameters={"symbol": "EURUSD", "timeframe": "H1"}
)

result = await compiled.ainvoke(state)
```

### Run with Streaming

```python
async for event in compiled.astream(state):
    for node, node_state in event.items():
        print(f"{node}: {node_state.get('stage')}")
```

### Enable Checkpointing

```python
from trading_orchestrator.state import GraphConfig

config = GraphConfig(enable_checkpointing=True)
orchestrator = TradingOrchestrator(llm, config)
compiled = await orchestrator.compile()

# Use thread_id for checkpoint persistence
config_dict = {"configurable": {"thread_id": "my-workflow-001"}}
result = await compiled.ainvoke(state, config_dict)

# Resume later
result = await compiled.ainvoke(None, config_dict)
```

## Agent Usage

### Data Engineer

```python
from trading_orchestrator.agents import DataEngineerAgent

agent = DataEngineerAgent(llm)
state.task_parameters = {
    "symbol": "EURUSD",
    "timeframe": "H1",
    "lookback_days": 730
}
result = await agent.invoke(state)
# Access: result["dataset"]
```

### SMC Analyst

```python
from trading_orchestrator.agents import SMCAnalystAgent

agent = SMCAnalystAgent(llm)
# Requires: state.dataset
result = await agent.invoke(state)
# Access: result["candidate_features"]
```

### ML Engineer

```python
from trading_orchestrator.agents import MLEngineerAgent

agent = MLEngineerAgent(llm)
# Requires: state.dataset, state.candidate_features
result = await agent.invoke(state)
# Access: result["best_model"], result["selected_features"]
```

### Risk Manager

```python
from trading_orchestrator.agents import RiskManagerAgent

agent = RiskManagerAgent(llm)
# Requires: state.best_model or state.filtered_signals
result = await agent.invoke(state)
# Access: result["risk_assessments"]
```

### Validation Agent

```python
from trading_orchestrator.agents import ValidationAgent

agent = ValidationAgent(llm)
# Requires: state.best_model
result = await agent.invoke(state)
# Access: result["final_validation"]
```

### Signal Evaluator

```python
from trading_orchestrator.agents import SignalEvaluatorAgent

agent = SignalEvaluatorAgent(llm)
# Requires: state.best_model
result = await agent.invoke(state)
# Access: result["generated_signals"], result["filtered_signals"]
```

## State Management

### Create Initial State

```python
from trading_orchestrator.state import AgentState, WorkflowStage

state = AgentState(
    workflow_id="my-workflow-001",
    stage=WorkflowStage.INITIALIZATION,
    task_description="Develop trading bot",
    task_parameters={"symbol": "EURUSD"},
    max_iterations=10
)
```

### Access State Fields

```python
# Dataset
if state.dataset:
    print(f"Symbol: {state.dataset.symbol}")
    print(f"Quality: {state.dataset.quality_score:.2%}")

# Model
if state.best_model:
    print(f"Sharpe: {state.best_model.metrics.sharpe_ratio:.2f}")
    print(f"Drawdown: {state.best_model.metrics.max_drawdown:.2%}")

# Signals
for signal in state.filtered_signals:
    print(f"{signal.direction}: {signal.confidence:.2%}")

# Messages
for msg in state.messages:
    print(f"[{msg.role.value}] {msg.content}")
```

## Configuration

### Basic Configuration

```python
from trading_orchestrator.state import GraphConfig

config = GraphConfig(
    enable_checkpointing=True,
    enable_streaming=True,
    enable_human_approval=False,
    max_agent_iterations=10
)
```

### Human-in-the-Loop

```python
config = GraphConfig(
    enable_human_approval=True,
    approval_stages=[
        WorkflowStage.MODEL_TRAINING,
        WorkflowStage.DEPLOYMENT
    ]
)
```

### Parallel Execution

```python
config = GraphConfig(
    enable_parallel_execution=True,
    max_concurrent_agents=3
)
```

## LLM Configuration

### Use Different Providers

```python
# Anthropic (default)
from trading_orchestrator.llm_factory import create_llm
llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# OpenAI
llm = create_llm(provider="openai", model="gpt-4-turbo-preview")

# Fast model for simple tasks
from trading_orchestrator.llm_factory import create_fast_llm
fast_llm = create_fast_llm()

# Smart model for complex tasks
from trading_orchestrator.llm_factory import create_smart_llm
smart_llm = create_smart_llm()
```

## Common Patterns

### Parallel Agent Execution

```python
import asyncio
from trading_orchestrator.agents import SMCAnalystAgent, MLEngineerAgent

smc_analyst = SMCAnalystAgent(llm)
ml_engineer = MLEngineerAgent(llm)

results = await asyncio.gather(
    smc_analyst.invoke(state),
    ml_engineer.invoke(state),
    return_exceptions=True
)

# Merge results
for result in results:
    if not isinstance(result, Exception):
        state = AgentState(**{**state.model_dump(), **result})
```

### Error Handling

```python
try:
    result = await agent.invoke(state)
except Exception as e:
    print(f"Agent failed: {e}")
    state.errors.append(str(e))

# Check for errors in state
if state.errors:
    print("Errors occurred:")
    for error in state.errors:
        print(f"  - {error}")
```

### Custom Agent Creation

```python
from trading_orchestrator.agents.base import BaseAgent
from trading_orchestrator.state import AgentRole

class CustomAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm, AgentRole.SUPERVISOR, "CustomAgent")

    @property
    def system_prompt(self):
        return "You are a custom agent..."

    async def _execute(self, state):
        # Your logic here
        result = await self._call_llm(
            self.system_prompt,
            "Analyze this state..."
        )

        return {
            "message": result,
            "state_updates": {"field": "value"}
        }
```

## Testing

### Run Tests

```bash
# All tests
poetry run pytest

# Specific test file
poetry run pytest tests/test_agents.py

# With coverage
poetry run pytest --cov=trading_orchestrator

# Verbose output
poetry run pytest -v
```

### Mock LLM for Testing

```python
from unittest.mock import AsyncMock, MagicMock

mock_llm = MagicMock()
mock_llm.ainvoke = AsyncMock(
    return_value=MagicMock(content="Test response")
)

agent = DataEngineerAgent(mock_llm)
```

## Debugging

### Enable Debug Logging

```python
# In .env
LOG_LEVEL=DEBUG
LOG_FORMAT=console

# Or programmatically
from trading_orchestrator.config import settings
settings.log_level = "DEBUG"
```

### Inspect State at Each Step

```python
async for event in compiled.astream(state):
    for node_name, node_state in event.items():
        print(f"\n{'='*60}")
        print(f"Node: {node_name}")
        print(f"Stage: {node_state.get('stage')}")
        print(f"Iteration: {node_state.get('iteration')}")

        # Inspect full state
        import json
        print(json.dumps(node_state, indent=2, default=str))
```

### View Checkpoint Data

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import sqlite3

# View checkpoints
conn = sqlite3.connect("checkpoints/checkpoints.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT 10")
for row in cursor.fetchall():
    print(row)
```

## Deployment

### Docker

```bash
# Build
docker build -t trading-orchestrator .

# Run
docker run -d \
  --name trading-orchestrator \
  -e ANTHROPIC_API_KEY=your_key \
  -v $(pwd)/checkpoints:/app/checkpoints \
  trading-orchestrator
```

### Docker Compose

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f orchestrator

# Stop
docker-compose down
```

### Kubernetes

```bash
# Deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n trading-orchestrator

# View logs
kubectl logs -f deployment/trading-orchestrator -n trading-orchestrator
```

## Monitoring

### View Logs

```bash
# Local
tail -f logs/orchestrator.log

# Docker
docker logs -f trading-orchestrator

# Kubernetes
kubectl logs -f deployment/trading-orchestrator -n trading-orchestrator
```

### Check Metrics

```python
# In code
from trading_orchestrator.logging_config import get_logger

logger = get_logger("metrics")
logger.info("workflow_completed", duration=120.5, success=True)
```

## Common Issues

### API Rate Limits

```python
# Implement retry with backoff
import asyncio

for attempt in range(3):
    try:
        result = await llm.ainvoke(messages)
        break
    except RateLimitError:
        if attempt == 2:
            raise
        await asyncio.sleep(2 ** attempt)
```

### Memory Issues

```python
# Reduce batch size
config = GraphConfig(max_concurrent_agents=1)

# Or clear old messages
state.messages = list(state.messages[-10:])  # Keep last 10
```

### Checkpoint Recovery

```bash
# If checkpoint is corrupted, delete and restart
rm checkpoints/checkpoints.db

# Or manually fix in SQLite
sqlite3 checkpoints/checkpoints.db
> DELETE FROM checkpoints WHERE thread_id = 'problematic-id';
```

## Environment Variables

```bash
# LLM Configuration
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_TEMPERATURE=0.1

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_orchestrator

# Agent Settings
MAX_AGENT_ITERATIONS=10
AGENT_TIMEOUT_SECONDS=300
ENABLE_HUMAN_IN_LOOP=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Monitoring
ENABLE_METRICS=false
METRICS_PORT=9090
```

## CLI Commands

```bash
# Run workflow
poetry run python -m trading_orchestrator.cli

# Run example
poetry run python examples/basic_workflow.py

# Run with checkpointing
poetry run python examples/checkpointing_demo.py

# Resume checkpoint
poetry run python examples/checkpointing_demo.py resume <workflow_id>

# Run tests
poetry run pytest

# Format code
poetry run black trading_orchestrator/

# Type check
poetry run mypy trading_orchestrator/
```

## Performance Tips

1. **Use caching** for repeated queries
2. **Enable parallel execution** for independent tasks
3. **Use fast LLM** for simple operations
4. **Implement batching** where possible
5. **Monitor resource usage** and adjust limits
6. **Use streaming** for long-running workflows
7. **Checkpoint frequently** for expensive operations
8. **Clean up old checkpoints** to save storage

## Best Practices

1. **Always validate state** before agent execution
2. **Use strongly-typed state models** (Pydantic)
3. **Implement comprehensive error handling**
4. **Add detailed logging** at decision points
5. **Test error scenarios** explicitly
6. **Document custom agents** thoroughly
7. **Use descriptive workflow IDs**
8. **Monitor performance metrics**
9. **Implement proper cleanup** (connections, resources)
10. **Keep dependencies updated**

## Resources

- **Documentation**: README.md, ARCHITECTURE.md, DEPLOYMENT.md
- **Examples**: examples/ directory
- **Tests**: tests/ directory
- **Issues**: GitHub Issues
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangChain Docs**: https://python.langchain.com/

## Quick Commands Reference

```bash
# Setup
./quickstart.sh                    # Initial setup

# Development
poetry install                     # Install dependencies
poetry run pytest                  # Run tests
poetry run black .                 # Format code
poetry run mypy trading_orchestrator/  # Type check

# Execution
poetry run python -m trading_orchestrator.cli  # Run CLI
poetry run python examples/basic_workflow.py   # Run example

# Docker
docker-compose up -d               # Start services
docker-compose logs -f             # View logs
docker-compose down                # Stop services

# Kubernetes
kubectl apply -f k8s/              # Deploy
kubectl get pods -n trading-orchestrator  # Check status
kubectl logs -f deployment/trading-orchestrator  # View logs
```
