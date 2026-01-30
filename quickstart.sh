#!/bin/bash

# Quickstart script for Trading Orchestrator
# This script helps you get up and running quickly

set -e

echo "================================================"
echo "Trading Orchestrator - Quick Start"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher is required"
    echo "   Current version: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Check for Poetry
echo "Checking for Poetry..."
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ Poetry installed"
echo ""

# Install dependencies
echo "Installing dependencies..."
poetry install
echo "✓ Dependencies installed"
echo ""

# Setup environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
    echo ""
    read -p "Press Enter to open .env file in nano editor..."
    nano .env
else
    echo "✓ .env file already exists"
fi
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints data logs
echo "✓ Directories created"
echo ""

# Check API keys
echo "Validating API keys..."
source .env
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: No API key found in .env"
    echo "   Please add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env"
    exit 1
fi
echo "✓ API keys configured"
echo ""

# Run tests
echo "Running tests..."
if poetry run pytest -v; then
    echo "✓ All tests passed"
else
    echo "⚠️  Some tests failed, but continuing..."
fi
echo ""

# Success message
echo "================================================"
echo "✓ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Run a basic workflow:"
echo "   poetry run python examples/basic_workflow.py"
echo ""
echo "2. Try the CLI:"
echo "   poetry run python -m trading_orchestrator.cli"
echo ""
echo "3. Explore examples:"
echo "   poetry run python examples/parallel_execution.py"
echo "   poetry run python examples/checkpointing_demo.py"
echo ""
echo "4. Read the documentation:"
echo "   - README.md - Overview and usage"
echo "   - ARCHITECTURE.md - System architecture"
echo "   - DEPLOYMENT.md - Deployment guide"
echo ""
echo "================================================"
