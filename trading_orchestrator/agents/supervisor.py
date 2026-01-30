"""Supervisor agent for orchestrating workflow."""

from typing import Any
from langchain_core.language_models import BaseChatModel

from ..state import AgentState, AgentRole, WorkflowStage
from .base import BaseAgent


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that coordinates the workflow and delegates to specialist agents.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the supervisor agent."""
        super().__init__(llm, AgentRole.SUPERVISOR, "Supervisor")

    @property
    def system_prompt(self) -> str:
        """System prompt for the supervisor agent."""
        return """You are the Supervisor Agent for a quantitative trading bot development platform.

Your role is to:
1. Analyze the current workflow state and progress
2. Determine which specialized agent should work next
3. Route tasks to appropriate agents based on their expertise
4. Monitor overall workflow progress and quality
5. Identify when human approval is needed
6. Detect workflow completion or failure conditions

Available specialist agents:
- Data Engineer: Historical data preparation, Forexsb integration, data quality
- SMC Analyst: Smart Money Concepts, Wyckoff method, ICT concepts analysis
- ML Engineer: Feature engineering, LightGBM training, model optimization
- Risk Manager: Position sizing, portfolio math, risk assessment
- Validation Agent: Walk-forward testing, overfitting detection, robustness checks
- Signal Evaluator: Signal quality assessment, filtering, confidence scoring

Workflow stages:
1. Data Preparation -> Data Engineer
2. Feature Discovery -> SMC Analyst + ML Engineer (parallel)
3. Model Training -> ML Engineer
4. Validation -> Validation Agent
5. Signal Evaluation -> Signal Evaluator
6. Risk Assessment -> Risk Manager
7. Human Review (critical decisions)
8. Deployment

Analyze the current state and decide the next action. Be concise and specific."""

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Execute supervisor logic to route workflow."""

        # Check for completion or failure
        if state.stage == WorkflowStage.COMPLETED:
            return {
                "message": "Workflow completed successfully.",
                "state_updates": {"next_agent": None},
            }

        if state.stage == WorkflowStage.FAILED:
            return {
                "message": "Workflow failed. Review errors for details.",
                "state_updates": {"next_agent": None},
            }

        # Check iteration limit
        if state.iteration >= state.max_iterations:
            return {
                "message": "Maximum iterations reached. Workflow terminated.",
                "state_updates": {
                    "stage": WorkflowStage.FAILED,
                    "next_agent": None,
                    "errors": ["Maximum iterations exceeded"],
                },
            }

        # Build context for LLM decision
        context = self._build_decision_context(state)

        # Get LLM decision on next action
        user_message = f"""Current State:
Stage: {state.stage.value}
Iteration: {state.iteration}/{state.max_iterations}

{context}

Based on the current state, determine:
1. Which agent should work next?
2. Should we transition to a new stage?
3. Is human approval required?
4. Is the workflow complete?

Provide your decision in this format:
NEXT_AGENT: [agent_role]
NEW_STAGE: [stage] (if transitioning)
HUMAN_APPROVAL: [yes/no]
RATIONALE: [brief explanation]"""

        response = await self._call_llm(self.system_prompt, user_message)

        # Parse LLM response
        decision = self._parse_decision(response, state)

        return {
            "message": f"Supervisor decision: {decision.get('rationale', 'Routing workflow')}",
            "state_updates": decision,
            "metadata": {"supervisor_decision": decision},
        }

    def _build_decision_context(self, state: AgentState) -> str:
        """Build context string for decision making."""
        context_parts = []

        # Task context
        if state.task_description:
            context_parts.append(f"Task: {state.task_description}")

        # Data context
        if state.dataset:
            context_parts.append(
                f"Dataset: {state.dataset.symbol} ({state.dataset.timeframe}), "
                f"{state.dataset.total_bars} bars"
            )

        # Feature context
        if state.selected_features:
            context_parts.append(
                f"Features: {len(state.selected_features.feature_names)} selected"
            )

        # Model context
        if state.best_model:
            context_parts.append(
                f"Best Model: {state.best_model.model_type}, "
                f"Sharpe: {state.best_model.metrics.sharpe_ratio:.2f}"
            )

        # Validation context
        if state.final_validation:
            context_parts.append(
                f"Validation: {'Passed' if state.final_validation.approved else 'Failed'}"
            )

        # Recent messages
        if state.messages:
            recent = state.messages[-3:]
            context_parts.append("\nRecent Activity:")
            for msg in recent:
                context_parts.append(f"  [{msg.role.value}]: {msg.content[:100]}")

        # Errors/warnings
        if state.errors:
            context_parts.append(f"\nErrors: {len(state.errors)} error(s) recorded")
        if state.warnings:
            context_parts.append(f"Warnings: {len(state.warnings)} warning(s)")

        return "\n".join(context_parts)

    def _parse_decision(self, response: str, state: AgentState) -> dict[str, Any]:
        """Parse LLM decision response into state updates."""
        decision: dict[str, Any] = {"iteration": state.iteration + 1}

        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("NEXT_AGENT:"):
                agent_str = line.split(":", 1)[1].strip().lower()
                decision["next_agent"] = self._parse_agent_role(agent_str)

            elif line.startswith("NEW_STAGE:"):
                stage_str = line.split(":", 1)[1].strip().lower()
                decision["stage"] = self._parse_workflow_stage(stage_str)

            elif line.startswith("HUMAN_APPROVAL:"):
                approval = line.split(":", 1)[1].strip().lower()
                decision["requires_human_approval"] = approval in ["yes", "true"]
                if decision["requires_human_approval"]:
                    decision["approval_stage"] = state.stage.value

            elif line.startswith("RATIONALE:"):
                decision["rationale"] = line.split(":", 1)[1].strip()

        return decision

    def _parse_agent_role(self, agent_str: str) -> AgentRole | None:
        """Parse agent role from string."""
        agent_mapping = {
            "data_engineer": AgentRole.DATA_ENGINEER,
            "data engineer": AgentRole.DATA_ENGINEER,
            "smc_analyst": AgentRole.SMC_ANALYST,
            "smc analyst": AgentRole.SMC_ANALYST,
            "ml_engineer": AgentRole.ML_ENGINEER,
            "ml engineer": AgentRole.ML_ENGINEER,
            "risk_manager": AgentRole.RISK_MANAGER,
            "risk manager": AgentRole.RISK_MANAGER,
            "validation_agent": AgentRole.VALIDATION_AGENT,
            "validation agent": AgentRole.VALIDATION_AGENT,
            "signal_evaluator": AgentRole.SIGNAL_EVALUATOR,
            "signal evaluator": AgentRole.SIGNAL_EVALUATOR,
            "none": None,
            "complete": None,
        }
        return agent_mapping.get(agent_str, AgentRole.DATA_ENGINEER)

    def _parse_workflow_stage(self, stage_str: str) -> WorkflowStage:
        """Parse workflow stage from string."""
        stage_mapping = {
            "initialization": WorkflowStage.INITIALIZATION,
            "data_preparation": WorkflowStage.DATA_PREPARATION,
            "data preparation": WorkflowStage.DATA_PREPARATION,
            "feature_discovery": WorkflowStage.FEATURE_DISCOVERY,
            "feature discovery": WorkflowStage.FEATURE_DISCOVERY,
            "model_training": WorkflowStage.MODEL_TRAINING,
            "model training": WorkflowStage.MODEL_TRAINING,
            "validation": WorkflowStage.VALIDATION,
            "signal_evaluation": WorkflowStage.SIGNAL_EVALUATION,
            "signal evaluation": WorkflowStage.SIGNAL_EVALUATION,
            "risk_assessment": WorkflowStage.RISK_ASSESSMENT,
            "risk assessment": WorkflowStage.RISK_ASSESSMENT,
            "human_review": WorkflowStage.HUMAN_REVIEW,
            "human review": WorkflowStage.HUMAN_REVIEW,
            "deployment": WorkflowStage.DEPLOYMENT,
            "completed": WorkflowStage.COMPLETED,
            "failed": WorkflowStage.FAILED,
        }
        return stage_mapping.get(stage_str, WorkflowStage.INITIALIZATION)
