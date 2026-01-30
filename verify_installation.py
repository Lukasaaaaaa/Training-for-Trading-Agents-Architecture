#!/usr/bin/env python3
"""
Installation verification script for Trading Orchestrator.

Run this after installation to verify everything is set up correctly.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version is 3.11+"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False


def check_imports():
    """Check required imports"""
    print("\nChecking package imports...")

    imports_to_check = [
        ("trading_orchestrator", "Main package"),
        ("trading_orchestrator.config", "Configuration"),
        ("trading_orchestrator.state", "State management"),
        ("trading_orchestrator.graph", "Graph orchestration"),
        ("trading_orchestrator.agents", "Agent implementations"),
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("pydantic", "Pydantic"),
        ("structlog", "Structlog"),
    ]

    all_passed = True
    for module, description in imports_to_check:
        try:
            __import__(module)
            print(f"  ✓ {description} ({module})")
        except ImportError as e:
            print(f"  ✗ {description} ({module}): {e}")
            all_passed = False

    return all_passed


def check_environment():
    """Check environment setup"""
    print("\nChecking environment configuration...")

    env_file = Path(".env")
    if env_file.exists():
        print("  ✓ .env file exists")

        # Check for API keys (without exposing values)
        env_content = env_file.read_text()
        has_anthropic = "ANTHROPIC_API_KEY=" in env_content and "your_" not in env_content
        has_openai = "OPENAI_API_KEY=" in env_content and "your_" not in env_content

        if has_anthropic or has_openai:
            print("  ✓ API key(s) configured")
            return True
        else:
            print("  ⚠ No API keys found in .env (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")
            return False
    else:
        print("  ✗ .env file not found (copy from .env.example)")
        return False


def check_directories():
    """Check required directories exist"""
    print("\nChecking directories...")

    directories = ["checkpoints", "data", "logs"]
    all_exist = True

    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"  ✓ {directory}/ exists")
        else:
            print(f"  ⚠ {directory}/ not found (will be created on first run)")

    return True  # Not critical


def check_agents():
    """Check all agents can be imported"""
    print("\nChecking agent implementations...")

    try:
        from trading_orchestrator.agents import (
            SupervisorAgent,
            DataEngineerAgent,
            SMCAnalystAgent,
            MLEngineerAgent,
            RiskManagerAgent,
            ValidationAgent,
            SignalEvaluatorAgent,
        )

        agents = [
            ("SupervisorAgent", SupervisorAgent),
            ("DataEngineerAgent", DataEngineerAgent),
            ("SMCAnalystAgent", SMCAnalystAgent),
            ("MLEngineerAgent", MLEngineerAgent),
            ("RiskManagerAgent", RiskManagerAgent),
            ("ValidationAgent", ValidationAgent),
            ("SignalEvaluatorAgent", SignalEvaluatorAgent),
        ]

        for name, agent_class in agents:
            print(f"  ✓ {name}")

        return True

    except ImportError as e:
        print(f"  ✗ Agent import failed: {e}")
        return False


def check_state_models():
    """Check state models"""
    print("\nChecking state models...")

    try:
        from trading_orchestrator.state import (
            AgentState,
            WorkflowStage,
            AgentRole,
            DatasetInfo,
            FeatureSet,
            ModelConfig,
            TradingSignal,
            RiskAssessment,
            ValidationResult,
        )

        models = [
            "AgentState",
            "WorkflowStage",
            "AgentRole",
            "DatasetInfo",
            "FeatureSet",
            "ModelConfig",
            "TradingSignal",
            "RiskAssessment",
            "ValidationResult",
        ]

        for model in models:
            print(f"  ✓ {model}")

        return True

    except ImportError as e:
        print(f"  ✗ State model import failed: {e}")
        return False


def check_examples():
    """Check example files exist"""
    print("\nChecking examples...")

    examples = [
        "examples/basic_workflow.py",
        "examples/parallel_execution.py",
        "examples/checkpointing_demo.py",
    ]

    all_exist = True
    for example in examples:
        path = Path(example)
        if path.exists():
            print(f"  ✓ {example}")
        else:
            print(f"  ✗ {example} not found")
            all_exist = False

    return all_exist


def run_basic_test():
    """Run a basic instantiation test"""
    print("\nRunning basic functionality test...")

    try:
        from trading_orchestrator.state import AgentState, WorkflowStage

        # Create a basic state
        state = AgentState(
            workflow_id="test-001",
            stage=WorkflowStage.INITIALIZATION,
            task_description="Test workflow",
        )

        print(f"  ✓ Created AgentState: {state.workflow_id}")
        print(f"  ✓ Stage: {state.stage.value}")
        print(f"  ✓ State validation passed")

        return True

    except Exception as e:
        print(f"  ✗ Basic test failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Trading Orchestrator - Installation Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Environment", check_environment),
        ("Directories", check_directories),
        ("Agents", check_agents),
        ("State Models", check_state_models),
        ("Examples", check_examples),
        ("Basic Functionality", run_basic_test),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print("-" * 60)
    print(f"Result: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All checks passed! Installation verified successfully.")
        print("\nNext steps:")
        print("  1. Run: poetry run python examples/basic_workflow.py")
        print("  2. Or:  poetry run python -m trading_orchestrator.cli")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        print("\nCommon fixes:")
        print("  - Run: poetry install")
        print("  - Copy: cp .env.example .env")
        print("  - Edit .env with your API keys")
        print("  - Run: mkdir -p checkpoints data logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
