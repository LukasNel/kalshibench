#!/bin/bash
set -e

# Parse arguments
OUTPUT_DIR="${1:-}"

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

# Build command
CMD="python run_neurips_benchmark.py --test"
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

# Run with uv (or fall back to python)
if command -v uv &> /dev/null; then
    uv run $CMD
else
    $CMD
fi

