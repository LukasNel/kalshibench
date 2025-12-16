#!/usr/bin/env python3
"""
KalshiBench NeurIPS Paper Evaluation Script

This script runs a comprehensive evaluation across multiple model families
for inclusion in a NeurIPS Datasets & Benchmarks submission.

Estimated costs (at 300 samples):
  - Tier 1 (Flagships):     ~$25-35
  - Tier 2 (Mid-tier):      ~$8-12
  - Tier 3 (Reasoning):     ~$20-30
  - Tier 4 (Budget/Small):  ~$2-4
  Total estimated: ~$55-80

Run time estimate: 2-4 hours depending on rate limits
"""

import subprocess
import sys
import os
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
    # Tier 1: Flagship models (best from each provider)
    "tier1_flagships": [
        "gpt-5.2",
        "claude-opus-4.5",
        # "gemini-2.0-flash",
        "llama-3.1-405b",
    ],
    
    # Tier 2: Mid-tier production models
    "tier2_midtier": [
        "gpt-4o",
        "claude-sonnet-4.5",
        "claude-3-5-sonnet",
        "qwen-2.5-72b",
        "llama-3.3-70b",
        "deepseek-v3",
        "mistral-large",
    ],
    
    # Tier 3: Reasoning-specialized models (key comparison for paper)
    "tier3_reasoning": [
        "o1",
        "o1-mini",
        "o3-mini",
        "qwen-qwq-32b",
        "deepseek-r1",
    ],
    
    # Tier 4: Budget/smaller models (scaling analysis)
    "tier4_budget": [
        "gpt-4o-mini",
        "claude-3-5-haiku",
        "qwen-2.5-7b",
        "llama-3.1-8b",
    ],
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
        "OPENAI_API_KEY": ["gpt-5.2", "gpt-5.1", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
        "ANTHROPIC_API_KEY": ["claude-opus-4.5", "claude-sonnet-4.5", "claude-3-5-sonnet", "claude-3-5-haiku"],
        "TOGETHER_API_KEY": ["qwen-2.5-72b", "qwen-2.5-7b", "qwen-qwq-32b", "llama-3.3-70b", "llama-3.1-405b", 
                            "llama-3.1-8b", "deepseek-v3", "deepseek-r1", "mistral-large"],
        "GEMINI_API_KEY": ["gemini-2.0-flash", "gemini-1.5-pro"],
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


def run_tier(tier_name: str, models: list[str], samples: int, output_dir: str):
    """Run benchmark for a specific tier of models."""
    print(f"\n{'='*70}")
    print(f"Running {tier_name}: {', '.join(models)}")
    print(f"{'='*70}\n")
    
    tier_output = os.path.join(output_dir, tier_name)
    
    cmd = [
        sys.executable, "kalshibench.py",
        "--models", *models,
        "--samples", str(samples),
        "--output", tier_output,
        "--concurrent", "5",  # Conservative to avoid rate limits
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"⚠️  {tier_name} completed with errors")
        return False
    
    print(f"✓ {tier_name} completed successfully")
    return True


def run_full_benchmark(output_dir: str, num_samples: int = DEFAULT_NUM_SAMPLES):
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
    }
    import json
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run each tier
    results = {}
    for tier_name, models in MODEL_TIERS.items():
        success = run_tier(tier_name, models, num_samples, output_dir)
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


def run_single_tier(tier_name: str, output_dir: str, num_samples: int = DEFAULT_NUM_SAMPLES):
    """Run a single tier."""
    if tier_name not in MODEL_TIERS:
        print(f"Unknown tier: {tier_name}")
        print(f"Available: {', '.join(MODEL_TIERS.keys())}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    run_tier(tier_name, MODEL_TIERS[tier_name], num_samples, output_dir)


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

Tiers:
  tier1_flagships  - GPT-5.2, Claude Opus 4.5, Gemini, Llama 405B
  tier2_midtier    - GPT-4o, Claude Sonnet 4.5, Qwen 72B, etc.
  tier3_reasoning  - o1, o1-mini, o3-mini, QwQ, DeepSeek-R1
  tier4_budget     - GPT-4o-mini, Claude Haiku, small models
        """,
    )
    
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    parser.add_argument("--tier", type=str, help="Run specific tier")
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Override sample count")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: neurips_results_TIMESTAMP)")
    parser.add_argument("--list-tiers", action="store_true", help="List available tiers")
    
    args = parser.parse_args()
    
    # Get values from args
    num_samples = args.samples
    output_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
    
    if num_samples != DEFAULT_NUM_SAMPLES:
        print(f"Using {num_samples} samples per model")
    
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
        run_single_tier(args.tier, output_dir, num_samples)
    elif args.full:
        run_full_benchmark(output_dir, num_samples)
    else:
        parser.print_help()
        print("\n💡 Tip: Start with --test to verify setup, then run --full")

