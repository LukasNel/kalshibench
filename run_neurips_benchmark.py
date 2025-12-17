#!/usr/bin/env python3
"""
KalshiBench NeurIPS Paper Evaluation Script

This script runs a comprehensive evaluation across multiple model families
for inclusion in a NeurIPS Datasets & Benchmarks submission.

Estimated costs (at 300 samples):
  - Tier 1 (Flagships):     ~$35-50  (5 models: GPT-5.2-xhigh, Claude Opus, Qwen3-235B, DeepSeek-V3.2, Kimi-K2)
  - Tier 2 (Mid-tier):      ~$12-20  (10 models: GPT-4o, GPT-5.1, Claude Sonnet, Qwen, Llama, etc.)
  - Tier 3 (Reasoning):     ~$35-50  (8 models: GPT-5.2, GPT-5.2-high, o1, o3-mini, QwQ, DeepSeek-R1)
  - Tier 4 (Budget/Small):  ~$2-4    (5 models: GPT-5.2-low, GPT-4o-mini, Claude Haiku, small models)
  - Tier Google (optional): ~$5-10   (4 models: Gemini 3 Pro/Flash, 2.5/2.0 - uncomment if API key available)
  Total estimated: ~$85-125 (without Google)

Run time estimate: 3-5 hours depending on rate limits
"""

import subprocess
import sys
import os
import shutil
import glob
from datetime import datetime

# ==============================================================================
# Configuration
# ==============================================================================

# Number of samples - 300 gives good statistical power with reasonable cost
DEFAULT_NUM_SAMPLES = 300

# Output directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = f"neurips_results_{TIMESTAMP}"

# Model tiers for structured evaluation
MODEL_TIERS = {
    # Tier 1: Flagship models (best from each provider, max reasoning)
    "tier1_flagships": [
        "gpt-5.2-xhigh",        # OpenAI flagship with max reasoning
        "claude-opus-4.5",      # Anthropic flagship
        "qwen3-235b-thinking",  # Alibaba flagship (235B MoE, reasoning mode)
        "deepseek-v3.2",        # DeepSeek flagship (GPT-5 level, via Fireworks)
        "kimi-k2",              # Moonshot AI flagship (1T params, via Fireworks)
    ],
    
    # Tier 2: Mid-tier production models
    "tier2_midtier": [
        "gpt-4o",
        "gpt-5.1",              # GPT-5.1 (default reasoning=none)
        "claude-sonnet-4.5",
        "claude-4-5-sonnet",
        "qwen3-235b",           # Qwen3-235B Instruct (non-thinking)
        # "qwen3-32b",
        # "qwen-2.5-72b",
        # "llama-3.3-70b",
        # "llama-3.1-405b",
        # "deepseek-v3",          # DeepSeek V3 (via Fireworks)
        # "deepseek-v3.1",        # DeepSeek V3.1 (via Fireworks)
        # "mistral-large",
    ],
    
    # Tier 3: Reasoning-specialized models (key comparison for paper)
    "tier3_reasoning": [
        "gpt-5.2",              # GPT-5.2 with medium reasoning (default)
        # "gpt-5.2-high",         # GPT-5.2 with high reasoning
        # "gpt-5.1-high",         # GPT-5.1 with high reasoning
        "o1",
        # "o1-mini",
        # "o3-mini",
        # "qwen-qwq-32b",
        # "deepseek-r1",          # DeepSeek R1 reasoning model (via Fireworks)
    ],
    
    # Tier 4: Budget/smaller models (scaling analysis)
    "tier4_budget": [
        "gpt-5.2-low",          # GPT-5.2 with low reasoning
        "gpt-5-mini",           # GPT-5 Mini (small, efficient)
        "gpt-5-nano",           # GPT-5 Nano (smallest, most affordable)
        "gpt-4o-mini",
        "claude-4-5-haiku",
        "qwen-2.5-7b",
        "llama-3.1-8b",
    ],
    
    # # Tier: Google models (uncomment when GEMINI_API_KEY available)
    # "tier_google": [
    #     "gemini-3-pro",         # Google flagship
    #     "gemini-3-flash",       # Google fast
    #     "gemini-2.5-pro",       # Previous gen
    #     "gemini-2.0-flash",     # Budget
    # ],
}

# All models combined for full run
ALL_MODELS = []
for tier_models in MODEL_TIERS.values():
    ALL_MODELS.extend(tier_models)


# ==============================================================================
# Helper Functions
# ==============================================================================

def check_api_keys():
    """Check that required API keys are set."""
    required_keys = {
        "OPENAI_API_KEY": ["gpt-5.2", "gpt-5.2-low", "gpt-5.2-high", "gpt-5.2-xhigh",
                          "gpt-5.1", "gpt-5.1-medium", "gpt-5.1-high",
                          "gpt-5-mini", "gpt-5-nano",
                          "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
        "ANTHROPIC_API_KEY": ["claude-opus-4.5", "claude-sonnet-4.5", "claude-3-5-sonnet", "claude-3-5-haiku"],
        "TOGETHER_API_KEY": ["qwen3-235b", "qwen3-235b-thinking", "qwen3-32b", "qwen-2.5-72b", "qwen-2.5-7b", "qwen-qwq-32b",
                            "llama-3.3-70b", "llama-3.1-405b", "llama-3.1-8b", "mistral-large"],
        "FIREWORKS_API_KEY": ["deepseek-v3", "deepseek-v3.1", "deepseek-v3.2", "deepseek-r1", "kimi-k2"],
        "GEMINI_API_KEY": ["gemini-3-pro", "gemini-3-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    }
    
    missing = []
    for key, models in required_keys.items():
        if not os.environ.get(key) and not os.environ.get(key.replace("GEMINI", "GOOGLE")):
            # Check if any models from this provider are in our run
            if any(m in ALL_MODELS for m in models):
                missing.append(key)
    
    if missing:
        print("⚠️  Missing API keys:")
        for key in missing:
            print(f"   - {key}")
        print("\nSet them with:")
        for key in missing:
            print(f'   export {key}="your-key-here"')
        return False
    return True


def model_key_to_filename_prefix(model_key: str) -> str:
    """Convert model key to the filename prefix used in result files."""
    return model_key.lower().replace(' ', '_').replace('-', '_').replace('.', '_')


def find_previous_result(model_key: str, previous_run_dir: str) -> str | None:
    """
    Find a previous result file for a model in the given directory.
    Searches recursively in case results are in tier subdirectories.
    Returns the path to the file if found, None otherwise.
    """
    if not previous_run_dir or not os.path.isdir(previous_run_dir):
        return None
    
    prefix = model_key_to_filename_prefix(model_key)
    
    # Search recursively for matching files
    pattern = os.path.join(previous_run_dir, "**", f"{prefix}_*.json")
    matches = glob.glob(pattern, recursive=True)
    
    # Filter out summary/metadata files
    result_files = [
        f for f in matches 
    ]
    
    if result_files:
        # Return most recent if multiple matches
        return max(result_files, key=os.path.getmtime)
    
    return None


def copy_previous_results(models: list[str], previous_run_dir: str, target_dir: str) -> tuple[list[str], list[str]]:
    """
    Check for previous results and copy them to target directory.
    
    Returns:
        tuple of (models_to_run, models_skipped)
    """
    models_to_run = []
    models_skipped = []
    
    os.makedirs(target_dir, exist_ok=True)
    
    for model in models:
        prev_result = find_previous_result(model, previous_run_dir)
        if prev_result:
            # Copy to new location
            dest = os.path.join(target_dir, os.path.basename(prev_result))
            shutil.copy2(prev_result, dest)
            print(f"  ✓ Copied previous result for {model}")
            print(f"    {prev_result} → {dest}")
            models_skipped.append(model)
        else:
            models_to_run.append(model)
    
    return models_to_run, models_skipped


def run_tier(tier_name: str, models: list[str], samples: int, output_dir: str, previous_run_dir: str | None = None):
    """Run benchmark for a specific tier of models."""
    print(f"\n{'='*70}")
    print(f"Running {tier_name}: {', '.join(models)}")
    print(f"{'='*70}\n")
    
    tier_output = os.path.join(output_dir, tier_name)
    has_copied_results = False
    previous_run_tier_dir = os.path.join(previous_run_dir, tier_name)
    # Check for previous results to copy
    if previous_run_dir and os.path.isdir(previous_run_tier_dir):
        print(f"Checking for previous results in: {previous_run_tier_dir}")
        # just copy the entire directory
        shutil.copytree(previous_run_tier_dir, tier_output)
        print(f"Copied previous results to: {tier_output}")
        has_copied_results = True
    
    cmd = [
        sys.executable, "kalshibench.py",
        "--models", *models,  # Pass ALL models so they're included in the report
        "--samples", str(samples),
        "--output", tier_output,
        "--concurrent", "5",  # Conservative to avoid rate limits
    ]
    
    # If we copied some results, tell kalshibench to load them
    if has_copied_results:
        cmd.append("--load-existing")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"⚠️  {tier_name} completed with errors")
        raise Exception(f"Error running {tier_name}: {result.stderr}")
    
    print(f"✓ {tier_name} completed successfully")
    return True


def run_full_benchmark(output_dir: str, num_samples: int = DEFAULT_NUM_SAMPLES, previous_run_dir: str | None = None):
    """Run the complete NeurIPS benchmark."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   KalshiBench NeurIPS Evaluation                     ║
║                                                                      ║
║  Comprehensive LLM forecasting calibration benchmark                 ║
║  for NeurIPS Datasets & Benchmarks Track                            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration:")
    print(f"  Samples per model: {num_samples}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Total models:      {len(ALL_MODELS)}")
    print(f"  Model tiers:       {len(MODEL_TIERS)}")
    if previous_run_dir:
        print(f"  Previous run:      {previous_run_dir}")
    
    # Check API keys
    if not check_api_keys():
        print("\n❌ Please set missing API keys and re-run")
        sys.exit(1)
    
    print("\n✓ All API keys found")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run configuration
    config = {
        "timestamp": TIMESTAMP,
        "num_samples": num_samples,
        "model_tiers": MODEL_TIERS,
        "all_models": ALL_MODELS,
        "previous_run_dir": previous_run_dir,
    }
    import json
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run each tier
    results = {}
    for tier_name, models in MODEL_TIERS.items():
        success = run_tier(tier_name, models, num_samples, output_dir, previous_run_dir)
        results[tier_name] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    
    for tier_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {tier_name}")
    
    print(f"\nResults saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review results in each tier's summary.json")
    print("  2. Use paper_methods_prompt_*.txt to generate Methods section")
    print("  3. Use paper_results_prompt_*.txt to generate Results section")
    print("  4. Combine tier results for full paper analysis")


def run_quick_test(output_dir: str | None = None):
    """Run a quick test with minimal samples to verify setup."""
    print("Running quick test (10 samples, 2 models)...")
    
    test_models = ["gpt-4o-mini", "claude-3-5-haiku"]
    test_output = output_dir if output_dir else f"test_run_{TIMESTAMP}"
    
    cmd = [
        sys.executable, "kalshibench.py",
        "--models", *test_models,
        "--samples", "10",
        "--output", test_output,
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✓ Test passed! Results in {test_output}/")
        print("  Ready to run full benchmark with: python run_neurips_benchmark.py --full")
    else:
        print("\n✗ Test failed. Check API keys and dependencies.")


def run_single_tier(tier_name: str, output_dir: str, num_samples: int = DEFAULT_NUM_SAMPLES, previous_run_dir: str | None = None):
    """Run a single tier."""
    if tier_name not in MODEL_TIERS:
        print(f"Unknown tier: {tier_name}")
        print(f"Available: {', '.join(MODEL_TIERS.keys())}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    run_tier(tier_name, MODEL_TIERS[tier_name], num_samples, output_dir, previous_run_dir)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run KalshiBench for NeurIPS paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_neurips_benchmark.py --test                    # Quick test (2 models, 10 samples)
  python run_neurips_benchmark.py --full                    # Full benchmark (all tiers)
  python run_neurips_benchmark.py --full -o my_results      # Custom output directory
  python run_neurips_benchmark.py --tier tier1_flagships    # Single tier
  python run_neurips_benchmark.py --samples 500             # Override sample count
  python run_neurips_benchmark.py --full --previous-run ./old_results  # Resume/reuse previous results

Tiers:
  tier1_flagships  - GPT-5.2-xhigh, Claude Opus 4.5, Qwen3-235B, DeepSeek-V3.2, Kimi-K2
  tier2_midtier    - GPT-4o, GPT-5.1, Claude Sonnet 4.5, Qwen3-32B, Llama, etc.
  tier3_reasoning  - GPT-5.2 (medium), GPT-5.2-high, GPT-5.1-high, o1, o3-mini, QwQ, DeepSeek-R1
  tier4_budget     - GPT-5.2-low, GPT-4o-mini, Claude Haiku, small models
  tier_google      - (commented out) Gemini 3 Pro/Flash, Gemini 2.5/2.0
        """,
    )
    
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    parser.add_argument("--tier", type=str, help="Run specific tier")
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Override sample count")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: neurips_results_TIMESTAMP)")
    parser.add_argument("--previous-run", "-p", type=str, default=None, 
                        help="Previous run directory to reuse results from (skips models with existing results)")
    parser.add_argument("--list-tiers", action="store_true", help="List available tiers")
    
    args = parser.parse_args()
    
    # Get values from args
    num_samples = args.samples
    output_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
    previous_run_dir = args.previous_run
    if num_samples != DEFAULT_NUM_SAMPLES:
        print(f"Using {num_samples} samples per model")
    
    if previous_run_dir:
        if not os.path.isdir(previous_run_dir):
            print(f"❌ Previous run directory not found: {previous_run_dir}")
            sys.exit(1)
        print(f"Will reuse results from: {previous_run_dir}")
    
    if args.list_tiers:
        print("\nAvailable tiers:")
        for tier_name, models in MODEL_TIERS.items():
            print(f"\n  {tier_name}:")
            for m in models:
                print(f"    - {m}")
        sys.exit(0)
    
    if args.test:
        run_quick_test(output_dir)
    elif args.tier:
        run_single_tier(args.tier, output_dir, num_samples, previous_run_dir)
    elif args.full:
        run_full_benchmark(output_dir, num_samples, previous_run_dir)
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --test to verify setup, then run --full")

