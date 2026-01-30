# Architecture Documentation

## System Overview

The Quantitative Trading Bot Multi-Agent Orchestration System is built on LangGraph and implements a hierarchical supervisor pattern for coordinating specialized AI agents in automated trading bot development.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Client / API Layer                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Trading Orchestrator                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Supervisor Agent                            │  │
│  │  (Workflow Coordination & Routing Logic)                 │  │
│  └──────┬──────────────────────────────────────────┬────────┘  │
│         │                                           │            │
│  ┌──────▼──────────────────────────────────────────▼────────┐  │
│  │            LangGraph State Machine                       │  │
│  │  - State Management (Pydantic Models)                    │  │
│  │  - Node Execution Engine                                 │  │
│  │  - Edge Routing Logic                                    │  │
│  │  - Checkpointing System                                  │  │
│  └──────┬───────────────────────────────────────────────────┘  │
└─────────┼──────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────┐
│                    Specialized Agent Layer                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │   Data   │ │   SMC    │ │    ML    │ │   Risk   │           │
│  │ Engineer │ │ Analyst  │ │ Engineer │ │ Manager  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐                                      │
│  │Validation│ │  Signal  │                                      │
│  │  Agent   │ │Evaluator │                                      │
│  └──────────┘ └──────────┘                                      │
└─────────┬────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │     LLM      │  │   Database   │  │ Checkpoint   │          │
│  │   Provider   │  │  (Postgres)  │  │   Storage    │          │
│  │ (Anthropic/  │  │              │  │              │          │
│  │   OpenAI)    │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Trading Orchestrator

The main orchestration component that manages the entire workflow.

**Key Responsibilities:**
- Initialize and configure all specialized agents
- Build the LangGraph state machine
- Manage workflow execution and state transitions
- Handle checkpointing and recovery
- Route tasks to appropriate agents

**Core Components:**
```python
class TradingOrchestrator:
    - supervisor: SupervisorAgent
    - data_engineer: DataEngineerAgent
    - smc_analyst: SMCAnalystAgent
    - ml_engineer: MLEngineerAgent
    - risk_manager: RiskManagerAgent
    - validation_agent: ValidationAgent
    - signal_evaluator: SignalEvaluatorAgent
    - graph: StateGraph
    - config: GraphConfig
```

### 2. State Management

**AgentState** - Central state container with strongly-typed fields:

```python
AgentState:
    - workflow_id: str                    # Unique workflow identifier
    - stage: WorkflowStage                # Current workflow stage
    - current_agent: AgentRole            # Currently executing agent
    - next_agent: AgentRole               # Next agent to execute
    - iteration: int                      # Current iteration count
    - messages: Sequence[Message]         # Agent communication log
    - dataset: DatasetInfo                # Trading dataset metadata
    - candidate_features: List[FeatureSet] # Feature engineering candidates
    - selected_features: FeatureSet       # Final selected features
    - trained_models: List[ModelConfig]   # All trained models
    - best_model: ModelConfig             # Best performing model
    - generated_signals: List[TradingSignal] # Raw signals
    - filtered_signals: List[TradingSignal]  # Quality-filtered signals
    - risk_assessments: Dict[str, RiskAssessment] # Risk evaluations
    - validation_results: List[ValidationResult]  # Validation outcomes
    - requires_human_approval: bool       # HITL flag
    - errors: Sequence[str]               # Error accumulator
    - warnings: Sequence[str]             # Warning accumulator
    - checkpoints: Dict[str, str]         # Checkpoint metadata
```

**State Reducers:**
- `messages`: Uses `operator.add` to append new messages
- `errors`: Uses `operator.add` to accumulate errors
- `warnings`: Uses `operator.add` to accumulate warnings

### 3. Agent Architecture

All agents inherit from `BaseAgent` and implement the `_execute` method:

```python
class BaseAgent(ABC):
    - llm: BaseChatModel              # Language model instance
    - role: AgentRole                 # Agent's role identifier
    - name: str                       # Human-readable name
    - logger: Logger                  # Structured logger

    @abstractmethod
    async def _execute(state: AgentState) -> Dict[str, Any]:
        """Execute agent-specific logic"""
        pass

    async def invoke(state: AgentState) -> Dict[str, Any]:
        """Main entry point with error handling"""
        pass

    async def _call_llm(system: str, user: str) -> str:
        """Call LLM with prompts"""
        pass
```

#### Agent Specializations

**Data Engineer Agent:**
- Data acquisition and validation
- Forexsb integration
- Quality assessment
- Train/val/test splitting

**SMC Analyst Agent:**
- Smart Money Concepts analysis
- Wyckoff method application
- ICT concepts identification
- Market structure feature engineering

**ML Engineer Agent:**
- Technical indicator calculation
- Feature engineering and selection
- LightGBM training and optimization
- Model performance evaluation

**Risk Manager Agent:**
- Position sizing (Kelly criterion)
- Portfolio risk metrics (VaR, CVaR)
- Risk-reward validation
- Approval/rejection logic

**Validation Agent:**
- Walk-forward analysis
- Overfitting detection
- Statistical significance testing
- Robustness scoring

**Signal Evaluator Agent:**
- Signal generation from predictions
- Quality assessment
- Multi-filter application
- Signal prioritization

**Supervisor Agent:**
- Workflow stage management
- Agent routing decisions
- Human approval coordination
- Completion/failure detection

### 4. Graph Structure

```python
StateGraph:
    Nodes:
        - supervisor         # Orchestration hub
        - data_engineer      # Data preparation
        - smc_analyst        # SMC feature engineering
        - ml_engineer        # ML feature engineering & training
        - risk_manager       # Risk assessment
        - validation_agent   # Model validation
        - signal_evaluator   # Signal generation & filtering
        - human_review       # Human-in-the-loop gate

    Edges:
        supervisor -> [all specialist agents] (conditional)
        [all specialists] -> supervisor
        supervisor -> human_review (conditional)
        human_review -> supervisor | END (conditional)
        supervisor -> END (when complete/failed)
```

**Routing Logic:**

```python
def _route_from_supervisor(state: AgentState) -> AgentNode | END:
    if state.stage in [COMPLETED, FAILED]:
        return END

    if state.requires_human_approval and config.enable_human_approval:
        return "human_review"

    if state.next_agent is None:
        return END

    return agent_routing_map[state.next_agent]
```

### 5. Workflow Execution Flow

```
1. Initialization
   ├─> Create initial state
   ├─> Set task parameters
   └─> Start workflow

2. Supervisor Evaluates State
   ├─> Analyze current progress
   ├─> Determine next action
   └─> Route to specialist agent

3. Specialist Agent Executes
   ├─> Receive state
   ├─> Perform specialized task
   ├─> Update state with results
   └─> Return to supervisor

4. State Update & Checkpoint
   ├─> Merge state updates
   ├─> Increment iteration
   ├─> Save checkpoint (if enabled)
   └─> Continue to next step

5. Repeat 2-4 Until:
   ├─> Workflow completed successfully
   ├─> Human approval required
   ├─> Error encountered
   └─> Max iterations reached
```

## Data Flow

### Feature Discovery Phase (Parallel Execution)

```
┌─────────────────┐
│  Supervisor     │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Dataset │
    │  Ready  │
    └────┬────┘
         │
    ┌────▼────────────────────────┐
    │   Parallel Execution        │
    ├─────────────┬───────────────┤
    │             │               │
┌───▼──────┐  ┌──▼──────────┐   │
│   SMC    │  │     ML      │   │
│ Analyst  │  │  Engineer   │   │
│          │  │             │   │
│ Output:  │  │  Output:    │   │
│ - Order  │  │  - RSI      │   │
│   blocks │  │  - MACD     │   │
│ - FVG    │  │  - SMA      │   │
│ - BOS    │  │  - ATR      │   │
└───┬──────┘  └──┬──────────┘   │
    │             │               │
    └─────┬───────▼───────────────┘
          │
    ┌─────▼──────┐
    │  Combined  │
    │ Feature Set│
    └────────────┘
```

### Model Training Flow

```
┌─────────────────┐
│ Selected        │
│ Features        │
└────────┬────────┘
         │
    ┌────▼────────────┐
    │ ML Engineer     │
    │ - Hyperparams   │
    │ - Cross-val     │
    │ - Training      │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Trained Model   │
    │ + Metrics       │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Validation      │
    │ Agent           │
    │ - Walk-forward  │
    │ - Overfitting   │
    └────────┬────────┘
             │
        ┌────▼────┐
        │ Passed? │
        └─┬────┬──┘
      Yes │    │ No
    ┌─────▼┐  ┌▼────────────┐
    │Signal│  │Human Review │
    │Eval  │  │or Retry     │
    └──────┘  └─────────────┘
```

## Checkpointing Architecture

### Checkpoint Storage

```python
Checkpoint Structure:
    - checkpoint_id: str              # Unique checkpoint ID
    - workflow_id: str                # Associated workflow
    - timestamp: datetime             # Checkpoint creation time
    - state: AgentState               # Complete state snapshot
    - metadata: dict                  # Additional context

Storage Options:
    1. SQLite (Development)
       - File-based: checkpoints.db
       - Fast, simple, single-node

    2. PostgreSQL (Production)
       - Shared state across instances
       - ACID guarantees
       - Scalable

    3. Custom (Advanced)
       - Redis for distributed caching
       - S3 for long-term storage
```

### Checkpoint Lifecycle

```
1. State Update Occurs
   │
2. Check if Checkpointing Enabled
   ├─> Yes: Continue
   └─> No: Skip to state return

3. Serialize AgentState
   ├─> Pydantic .model_dump() conversion
   └─> JSON serialization

4. Write to Checkpoint Store
   ├─> Generate checkpoint_id
   ├─> Store with metadata
   └─> Update checkpoint index

5. Confirm Write Success
   ├─> Success: Continue workflow
   └─> Failure: Log error, continue (async write)
```

### Recovery Process

```
1. Detect Workflow Resumption
   ├─> thread_id provided
   └─> state is None

2. Load Latest Checkpoint
   ├─> Query by thread_id
   ├─> Order by timestamp DESC
   └─> Limit 1

3. Deserialize State
   ├─> Parse JSON
   └─> Reconstruct Pydantic models

4. Resume Execution
   ├─> Continue from current stage
   ├─> Use last next_agent value
   └─> Increment iteration

5. Continue Normal Flow
```

## Error Handling Strategy

### Error Categories

1. **Recoverable Errors**
   - LLM timeout → Retry with exponential backoff
   - Rate limit → Wait and retry
   - Network error → Retry with backoff

2. **Agent Errors**
   - Invalid output → Log, continue with degraded state
   - Exception in _execute → Catch, log, add to errors list

3. **Fatal Errors**
   - Database connection lost → Checkpoint, fail workflow
   - Invalid state transition → Log, mark workflow as failed
   - Max iterations exceeded → Complete with partial results

### Error Recovery Flow

```python
try:
    result = await agent.invoke(state)
except RecoverableError as e:
    # Retry logic
    for attempt in range(max_retries):
        try:
            result = await agent.invoke(state)
            break
        except RecoverableError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay * (2 ** attempt))

except FatalError as e:
    # Checkpoint and fail
    await checkpoint_state(state)
    state.stage = WorkflowStage.FAILED
    state.errors.append(str(e))
    return state

except Exception as e:
    # Unknown error - log and continue with degraded state
    logger.error("unexpected_error", error=str(e))
    state.errors.append(f"Unexpected error in {agent.name}: {str(e)}")
    state.warnings.append("Workflow continuing with degraded state")
    return state
```

## Performance Considerations

### Parallel Execution

The system supports parallel execution for independent tasks:

```python
# SMC Analyst and ML Engineer can run in parallel
# during feature discovery phase

parallel_results = await asyncio.gather(
    smc_analyst.invoke(state),
    ml_engineer.invoke(state),
    return_exceptions=True
)

# Merge results back into state
for result in parallel_results:
    if isinstance(result, Exception):
        handle_error(result)
    else:
        state.update(result)
```

### Streaming

Streaming provides real-time progress updates:

```python
async for event in compiled_graph.astream(initial_state):
    for node_name, node_state in event.items():
        # Process partial state updates in real-time
        emit_progress_update(node_name, node_state)
```

### Caching

Implement LLM response caching for repeated queries:

```python
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Identical prompts will return cached results
# Reduces API costs and latency
```

## Security Architecture

### API Key Management

```python
# Environment variables (development)
ANTHROPIC_API_KEY=sk-...

# Secrets manager (production)
from boto3 import client
secrets = client('secretsmanager')
api_key = secrets.get_secret_value(SecretId='anthropic-api-key')
```

### Data Protection

- All sensitive data encrypted at rest
- TLS 1.3 for data in transit
- No PII in logs or checkpoints
- Database connection encryption enforced

### Access Control

```python
# Role-based access control
roles = {
    "admin": ["read", "write", "deploy", "delete"],
    "developer": ["read", "write"],
    "viewer": ["read"],
}

# Workflow-level permissions
workflow_permissions = {
    "user_id": ["workflows:read", "workflows:execute"],
}
```

## Scalability

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-orchestrator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-orchestrator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### State Sharding

For high-volume scenarios, implement state sharding:

```python
# Shard workflows by symbol/strategy
shard_key = f"{symbol}_{strategy_id}"
checkpoint_table = f"checkpoints_{hash(shard_key) % num_shards}"
```

## Monitoring & Observability

### Key Metrics

```python
metrics = {
    "workflow_duration_seconds": Histogram,
    "agent_execution_time_seconds": Histogram,
    "model_training_duration_seconds": Histogram,
    "validation_pass_rate": Gauge,
    "signal_generation_count": Counter,
    "error_count_by_agent": Counter,
    "llm_token_usage": Counter,
    "checkpoint_write_latency_seconds": Histogram,
}
```

### Distributed Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("workflow_execution"):
    with tracer.start_as_current_span("agent_execution", attributes={"agent": agent.name}):
        result = await agent.invoke(state)
```

## Future Enhancements

1. **Multi-Strategy Orchestration**
   - Parallel execution of multiple strategies
   - Portfolio-level optimization
   - Strategy correlation analysis

2. **Reinforcement Learning Integration**
   - Agent policy optimization
   - Adaptive routing based on performance
   - Dynamic hyperparameter tuning

3. **Real-time Trading Mode**
   - Live market data integration
   - Order execution interface
   - Position management

4. **Advanced Human-in-the-Loop**
   - Web-based approval interface
   - Slack/Teams integration
   - Approval workflow automation

5. **Enhanced Monitoring**
   - Custom Grafana dashboards
   - Anomaly detection
   - Predictive alerting
