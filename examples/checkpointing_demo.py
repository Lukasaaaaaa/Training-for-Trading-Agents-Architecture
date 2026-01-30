"""Example demonstrating checkpointing for long-running workflows."""

import asyncio
from pathlib import Path
from trading_orchestrator.llm_factory import create_llm
from trading_orchestrator.graph import TradingOrchestrator
from trading_orchestrator.state import GraphConfig
from trading_orchestrator.logging_config import configure_logging


async def demo_checkpointing() -> None:
    """
    Demonstrate workflow checkpointing and recovery.

    This example shows how to:
    1. Start a workflow with checkpointing enabled
    2. Interrupt it mid-execution
    3. Resume from the last checkpoint
    """

    configure_logging()

    # Ensure checkpoint directory exists
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Create LLM and orchestrator
    llm = create_llm()
    config = GraphConfig(
        enable_checkpointing=True,
        checkpoint_dir=str(checkpoint_dir),
        enable_human_approval=False,
    )

    orchestrator = TradingOrchestrator(llm, config)
    compiled_graph = await orchestrator.compile()

    # Define task
    task_description = "Develop trading bot with checkpointing enabled"
    task_parameters = {
        "symbol": "GBPJPY",
        "timeframe": "H1",
        "lookback_days": 365,
    }

    # Create initial state
    initial_state = orchestrator.create_initial_state(
        task_description=task_description,
        task_parameters=task_parameters,
    )

    workflow_id = initial_state.workflow_id
    print(f"Starting workflow with checkpointing: {workflow_id}")
    print(f"Checkpoint directory: {checkpoint_dir}\n")

    # Configure checkpoint
    config_dict = {"configurable": {"thread_id": workflow_id}}

    try:
        # Execute workflow with checkpointing
        event_count = 0
        async for event in compiled_graph.astream(initial_state, config_dict):
            event_count += 1

            for node_name, node_state in event.items():
                if node_name == "__start__":
                    continue

                print(f"[Checkpoint {event_count}] Node: {node_name}")
                print(f"  Stage: {node_state.get('stage', 'unknown')}")

                # Simulate interruption after a few steps (for demo purposes)
                if event_count == 3:
                    print("\n[SIMULATED INTERRUPTION]")
                    print("Workflow interrupted. State has been checkpointed.")
                    print(f"To resume, use thread_id: {workflow_id}\n")

                    # In production, you would resume with:
                    # compiled_graph.astream(None, config_dict)
                    return

        print("\nWorkflow completed successfully!")

    except KeyboardInterrupt:
        print(f"\n\nWorkflow interrupted by user.")
        print(f"Checkpoint saved. Resume with thread_id: {workflow_id}")
        print("\nTo resume:")
        print(f"  config = {{'configurable': {{'thread_id': '{workflow_id}'}}}}")
        print(f"  compiled_graph.astream(None, config)")


async def resume_workflow(thread_id: str) -> None:
    """
    Resume a previously checkpointed workflow.

    Args:
        thread_id: The workflow ID to resume
    """

    configure_logging()

    # Create orchestrator with checkpointing
    llm = create_llm()
    config = GraphConfig(enable_checkpointing=True)

    orchestrator = TradingOrchestrator(llm, config)
    compiled_graph = await orchestrator.compile()

    # Configure with existing thread_id
    config_dict = {"configurable": {"thread_id": thread_id}}

    print(f"Resuming workflow: {thread_id}\n")

    # Resume from checkpoint (pass None as state to resume)
    async for event in compiled_graph.astream(None, config_dict):
        for node_name, node_state in event.items():
            if node_name == "__start__":
                continue

            print(f"[Resumed] Node: {node_name}")
            print(f"  Stage: {node_state.get('stage', 'unknown')}")

    print("\nWorkflow completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        if len(sys.argv) < 3:
            print("Usage: python checkpointing_demo.py resume <thread_id>")
            sys.exit(1)

        thread_id = sys.argv[2]
        asyncio.run(resume_workflow(thread_id))
    else:
        asyncio.run(demo_checkpointing())
