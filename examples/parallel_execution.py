"""Example of parallel agent execution for feature discovery."""

import asyncio
from typing import Any
from trading_orchestrator.llm_factory import create_llm
from trading_orchestrator.agents import SMCAnalystAgent, MLEngineerAgent
from trading_orchestrator.state import AgentState, DatasetInfo, WorkflowStage
from trading_orchestrator.logging_config import configure_logging
from datetime import datetime, timedelta


async def parallel_feature_discovery() -> None:
    """
    Demonstrate parallel execution of SMC Analyst and ML Engineer
    for simultaneous feature engineering.
    """

    configure_logging()

    # Create LLM
    llm = create_llm()

    # Create agents
    smc_analyst = SMCAnalystAgent(llm)
    ml_engineer = MLEngineerAgent(llm)

    # Create mock state with dataset
    state = AgentState(
        workflow_id="parallel-demo-001",
        stage=WorkflowStage.FEATURE_DISCOVERY,
        task_description="Parallel feature engineering demonstration",
        dataset=DatasetInfo(
            symbol="GBPUSD",
            timeframe="H4",
            start_date=datetime.utcnow() - timedelta(days=730),
            end_date=datetime.utcnow(),
            total_bars=4380,
            train_bars=2628,
            validation_bars=876,
            test_bars=876,
            quality_score=0.87,
        ),
    )

    print("Starting parallel feature discovery...")
    print(f"Dataset: {state.dataset.symbol} {state.dataset.timeframe}")
    print(f"Total bars: {state.dataset.total_bars}\n")

    # Execute both agents in parallel
    start_time = asyncio.get_event_loop().time()

    results = await asyncio.gather(
        smc_analyst.invoke(state),
        ml_engineer.invoke(state),
        return_exceptions=True,
    )

    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time

    print(f"\nParallel execution completed in {duration:.2f} seconds\n")

    # Process results
    for i, result in enumerate(results):
        agent_name = "SMC Analyst" if i == 0 else "ML Engineer"

        if isinstance(result, Exception):
            print(f"[ERROR] {agent_name}: {str(result)}")
            continue

        print(f"{'='*60}")
        print(f"{agent_name} Results:")
        print(f"{'='*60}")

        if "state_updates" in result:
            updates = result["state_updates"]

            if "candidate_features" in updates:
                features = updates["candidate_features"][0]
                print(f"Features generated: {len(features.feature_names)}")
                print(f"Feature names: {', '.join(features.feature_names[:10])}...")

        if "message" in result:
            print(f"\nMessage: {result['message'][:300]}...")

        print()


if __name__ == "__main__":
    asyncio.run(parallel_feature_discovery())
