#!/bin/bash
set -e

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found"
    echo "Create one with:"
    echo "  OPENAI_API_KEY=sk-..."
    echo "  ANTHROPIC_API_KEY=sk-ant-..."
    echo "  TOGETHER_API_KEY=..."
    echo "  GEMINI_API_KEY=..."
    exit 1
fi

# Run full NeurIPS benchmark with uv (or fall back to python)
if command -v uv &> /dev/null; then
    uv run python run_neurips_benchmark.py --full "$@"
else
    python run_neurips_benchmark.py --full "$@"
fi

