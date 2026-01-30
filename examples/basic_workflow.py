"""Basic workflow example for the trading orchestrator."""

import asyncio
from trading_orchestrator.llm_factory import create_llm
from trading_orchestrator.graph import TradingOrchestrator
from trading_orchestrator.state import GraphConfig
from trading_orchestrator.logging_config import configure_logging


async def main() -> None:
    """Run a basic trading bot development workflow."""

    # Configure logging
    configure_logging()

    # Create LLM
    llm = create_llm()

    # Configure graph
    config = GraphConfig(
        enable_checkpointing=True,
        enable_streaming=True,
        enable_human_approval=False,  # Disable for automated run
    )

    # Create orchestrator
    orchestrator = TradingOrchestrator(llm, config)
    compiled_graph = await orchestrator.compile()

    # Define task
    task_description = "Develop a trading bot for EURUSD H1 timeframe using SMC and ICT concepts"
    task_parameters = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "lookback_days": 730,
        "use_forexsb": False,
        "target_sharpe": 1.5,
        "max_drawdown": 0.15,
    }

    # Create initial state
    initial_state = orchestrator.create_initial_state(
        task_description=task_description,
        task_parameters=task_parameters,
    )

    print(f"Starting workflow: {initial_state.workflow_id}")
    print(f"Task: {task_description}\n")

    # Execute workflow with streaming
    async for event in compiled_graph.astream(initial_state):
        for node_name, node_state in event.items():
            if node_name == "__start__":
                continue

            print(f"\n{'='*60}")
            print(f"Node: {node_name}")
            print(f"Stage: {node_state.get('stage', 'unknown')}")

            # Print latest message
            if node_state.get("messages"):
                latest_message = node_state["messages"][-1]
                print(f"\nAgent: {latest_message.role.value}")
                print(f"Message: {latest_message.content[:200]}...")

            # Print key metrics if available
            if node_state.get("dataset"):
                dataset = node_state["dataset"]
                print(f"\nDataset: {dataset.symbol} {dataset.timeframe}")
                print(f"Quality: {dataset.quality_score:.2%}")

            if node_state.get("best_model"):
                model = node_state["best_model"]
                print(f"\nModel Sharpe: {model.metrics.sharpe_ratio:.2f}")
                print(f"Max Drawdown: {model.metrics.max_drawdown:.2%}")

            if node_state.get("filtered_signals"):
                signals = node_state["filtered_signals"]
                print(f"\nFiltered Signals: {len(signals)}")

    print("\n" + "="*60)
    print("Workflow completed!")


if __name__ == "__main__":
    asyncio.run(main())
