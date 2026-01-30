"""Command-line interface for the trading orchestrator."""

import asyncio
import sys
from typing import Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .logging_config import configure_logging, get_logger
from .llm_factory import create_llm
from .graph import TradingOrchestrator
from .state import GraphConfig

console = Console()
logger = get_logger("cli")


async def run_workflow(
    task_description: str,
    task_parameters: dict[str, Any],
    stream: bool = True,
) -> None:
    """
    Run a complete workflow for trading bot development.

    Args:
        task_description: Description of the task
        task_parameters: Parameters for the task
        stream: Whether to stream progress updates
    """
    console.print(Panel.fit(
        f"[bold cyan]Trading Bot Development Workflow[/bold cyan]\n"
        f"Task: {task_description}",
        border_style="cyan"
    ))

    # Initialize orchestrator
    llm = create_llm()
    config = GraphConfig(
        enable_checkpointing=True,
        enable_streaming=stream,
        enable_human_approval=settings.enable_human_in_loop,
    )

    orchestrator = TradingOrchestrator(llm, config)
    compiled_graph = await orchestrator.compile()

    # Create initial state
    initial_state = orchestrator.create_initial_state(
        task_description=task_description,
        task_parameters=task_parameters,
    )

    console.print(f"\n[green]Workflow ID:[/green] {initial_state.workflow_id}")
    console.print(f"[green]Started at:[/green] {initial_state.started_at}\n")

    # Execute workflow
    if stream:
        await run_with_streaming(compiled_graph, initial_state)
    else:
        await run_without_streaming(compiled_graph, initial_state)


async def run_with_streaming(
    compiled_graph: Any,
    initial_state: Any,
) -> None:
    """Run workflow with real-time streaming updates."""

    try:
        async for event in compiled_graph.astream(initial_state):
            for node_name, node_state in event.items():
                if node_name == "__start__":
                    continue

                console.print(f"\n[bold yellow]→ {node_name.replace('_', ' ').title()}[/bold yellow]")

                # Display latest message
                if node_state.get("messages"):
                    latest_msg = node_state["messages"][-1]
                    console.print(f"[dim]{latest_msg.content[:200]}...[/dim]")

                # Display stage
                if node_state.get("stage"):
                    console.print(f"[cyan]Stage: {node_state['stage'].value}[/cyan]")

                # Check for errors
                if node_state.get("errors"):
                    for error in node_state["errors"][-3:]:
                        console.print(f"[red]Error: {error}[/red]")

        console.print("\n[bold green]✓ Workflow completed[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Workflow failed: {str(e)}[/bold red]")
        logger.error("workflow_failed", error=str(e))
        raise


async def run_without_streaming(
    compiled_graph: Any,
    initial_state: Any,
) -> None:
    """Run workflow without streaming (batch execution)."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing workflow...", total=None)

        try:
            final_state = await compiled_graph.ainvoke(initial_state)

            progress.update(task, completed=True)

            # Display final results
            display_final_results(final_state)

        except Exception as e:
            progress.update(task, description=f"[red]Failed: {str(e)}[/red]")
            logger.error("workflow_failed", error=str(e))
            raise


def display_final_results(final_state: Any) -> None:
    """Display final workflow results in a structured format."""

    console.print("\n[bold green]═══ Final Results ═══[/bold green]\n")

    # Workflow summary
    summary_table = Table(title="Workflow Summary", show_header=False)
    summary_table.add_column("Field", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Workflow ID", final_state.workflow_id)
    summary_table.add_row("Stage", final_state.stage.value)
    summary_table.add_row("Iterations", str(final_state.iteration))
    summary_table.add_row(
        "Duration",
        str(final_state.completed_at - final_state.started_at)
        if final_state.completed_at
        else "In Progress"
    )

    console.print(summary_table)

    # Dataset info
    if final_state.dataset:
        console.print("\n[bold cyan]Dataset Information[/bold cyan]")
        dataset_table = Table(show_header=False)
        dataset_table.add_column("Field", style="cyan")
        dataset_table.add_column("Value", style="white")

        dataset_table.add_row("Symbol", final_state.dataset.symbol)
        dataset_table.add_row("Timeframe", final_state.dataset.timeframe)
        dataset_table.add_row("Total Bars", str(final_state.dataset.total_bars))
        dataset_table.add_row("Quality Score", f"{final_state.dataset.quality_score:.2%}")

        console.print(dataset_table)

    # Model performance
    if final_state.best_model:
        console.print("\n[bold cyan]Model Performance[/bold cyan]")
        metrics = final_state.best_model.metrics

        metrics_table = Table(show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")

        metrics_table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        metrics_table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        metrics_table.add_row("Win Rate", f"{metrics.win_rate:.2%}")
        metrics_table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
        metrics_table.add_row("Total Trades", str(metrics.total_trades))

        console.print(metrics_table)

    # Signals
    if final_state.filtered_signals:
        console.print(f"\n[bold cyan]Generated Signals: {len(final_state.filtered_signals)}[/bold cyan]")

        signal_table = Table()
        signal_table.add_column("Direction", style="cyan")
        signal_table.add_column("Confidence", style="white")
        signal_table.add_column("R:R", style="white")
        signal_table.add_column("SMC", style="white")

        for signal in final_state.filtered_signals[:10]:  # Top 10
            signal_table.add_row(
                signal.direction.upper(),
                f"{signal.confidence:.2%}",
                f"{signal.risk_reward_ratio:.2f}",
                "✓" if signal.smc_alignment else "✗",
            )

        console.print(signal_table)

    # Errors and warnings
    if final_state.errors:
        console.print(f"\n[bold red]Errors ({len(final_state.errors)}):[/bold red]")
        for error in final_state.errors[-5:]:
            console.print(f"  • {error}")

    if final_state.warnings:
        console.print(f"\n[bold yellow]Warnings ({len(final_state.warnings)}):[/bold yellow]")
        for warning in final_state.warnings[-5:]:
            console.print(f"  • {warning}")


async def main() -> None:
    """Main CLI entry point."""

    # Configure logging
    configure_logging()

    # Example workflow
    task_description = "Develop trading bot for EURUSD H1 timeframe with SMC analysis"
    task_parameters = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "lookback_days": 730,  # 2 years
        "use_forexsb": False,
        "target_sharpe": 1.5,
        "max_drawdown": 0.15,
    }

    try:
        await run_workflow(
            task_description=task_description,
            task_parameters=task_parameters,
            stream=True,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Workflow interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
