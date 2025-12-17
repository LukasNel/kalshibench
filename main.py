from datetime import UTC, datetime
import os
from typing import Protocol, cast

import modal  # pyright: ignore[reportMissingImports]
kalshibench_volume = modal.Volume.from_name("kalshibench-volume", create_if_missing=True)

class _ModalRemoteFn(Protocol):
    def remote(self, *args: object, **kwargs: object) -> object: ...
OUTPUT_DIR = "/prediction-market-grpo"
app = modal.App(name="prediction-market-grpo")
volume = modal.Volume.from_name("prediction-market-grpo-volume", create_if_missing=True)
image = modal.Image.debian_slim().uv_pip_install("unsloth", "vllm", "datasets", "evaluate", "trl").uv_pip_install("wandb").add_local_python_source('training')
HOURS = 3600
@app.function(gpu="A100", image=image, secrets=[modal.Secret.from_dotenv(".env")], volumes={OUTPUT_DIR: volume}, timeout=23*HOURS)
def train_model():
    from training import main
    _result = cast(object, main(OUTPUT_DIR))

# --- Benchmark/test runner (runs uv sync + run_test.sh) ---
REPO_DIR = "/root/predictionmarket"


def _write_dotenv_from_env(path: str) -> None:
    """Materialize a .env file from injected env vars so run_test.sh can load it."""
    import os

    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TOGETHER_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",  # For HuggingFace dataset upload
        "FIREWORKS_API_KEY",
    ]
    lines: list[str] = []
    for k in keys:
        v = os.environ.get(k)
        if v:
            lines.append(f"{k}={v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


test_image = (
    modal.Image.debian_slim()
    .apt_install("bash")
    .uv_sync()
    .add_local_file("kalshibench.py", os.path.join(REPO_DIR, "kalshibench.py"))
    .add_local_file("create_kalshibench.py", os.path.join(REPO_DIR, "create_kalshibench.py"))
    .add_local_file("run_test.sh", os.path.join(REPO_DIR, "run_test.sh"))
    .add_local_file("run_full.sh", os.path.join(REPO_DIR, "run_full.sh"))
    .add_local_file("README.md", os.path.join(REPO_DIR, "README.md"))
    .add_local_file("run_neurips_benchmark.py", os.path.join(REPO_DIR, "run_neurips_benchmark.py"))
)

DATASET_OUTPUT_DIR = "/neurips_results"
DATASET_FOLDER_NAME = "kalshibench_v3_" + datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
DATASET_OUTPUT_PATH = os.path.join(DATASET_OUTPUT_DIR, DATASET_FOLDER_NAME)
PREVIOUS_RUN_DIR = "/neurips_results/kalshibench_v2_2025-12-17_20-11-23"
PREVIOUS_RUN_DIR = os.path.join(DATASET_OUTPUT_DIR, "kalshibench_v2_2025-12-17_20-11-23")
PREVIOUS_RUN_PATH = os.path.join(DATASET_OUTPUT_DIR, "kalshibench_v2_2025-12-17_20-54-58")


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=2 * HOURS,
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def run_neurips_test() -> None:
    import subprocess
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    _ = subprocess.run(["bash", "run_test.sh", DATASET_OUTPUT_PATH], cwd=REPO_DIR, check=True)


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=12 * HOURS,  # Full benchmark takes longer
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def run_neurips_full() -> None:
    """Run the full NeurIPS benchmark across all model tiers."""
    import subprocess
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    command = [
        "uv", "run", "python", "run_neurips_benchmark.py", "--full",
        "--output", DATASET_OUTPUT_PATH,
        "--tier", "tier1_flagships",
        # "--previous-run", PREVIOUS_RUN_DIR,
    ]
    print(f"Running command: {command}")
    _ = subprocess.run(command, cwd=REPO_DIR, check=True)


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=2 * HOURS,
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def create_kalshibench_dataset(
    output_path: str = DATASET_OUTPUT_DIR,
    upload_repo: str | None = "2084Collective/kalshibench-v2",
) -> None:
    """Create and optionally upload the KalshiBench dataset.
    
    Args:
        output_path: Local path to save the dataset
        upload_repo: HuggingFace repo ID to upload to (e.g., '2084Collective/kalshibench-v2')
    """
    import subprocess
    
    # Write .env for HuggingFace token if uploading
    _write_dotenv_from_env(os.path.join(REPO_DIR, ".env"))
    cmd = ["uv", "run", "python", "create_kalshibench.py", "--output", os.path.join(DATASET_OUTPUT_PATH, "dataset_creation")]
    
    if upload_repo:
        cmd.extend(["--upload", upload_repo])
    
    _ = subprocess.run(cmd, cwd=REPO_DIR, check=True)
    print(f"Dataset created at {output_path}")
    if upload_repo:
        print(f"Dataset uploaded to {upload_repo}")


# @app.local_entrypoint()
# def test():
#     """Run test benchmark (quick validation)."""
#     # _ = cast(_ModalRemoteFn, create_kalshibench_dataset).remote(
#     #     output_path=DATASET_OUTPUT_DIR,
#     # )
#     _ = cast(_ModalRemoteFn, run_neurips_test).remote()


@app.function(
    image=test_image,
    secrets=[modal.Secret.from_dotenv(".env")],
    timeout=1 * HOURS,
    volumes={DATASET_OUTPUT_DIR: kalshibench_volume},
)
def generate_dataset_stats(
    dataset_name: str = "2084Collective/kalshibench-v2",
    output_path: str | None = None,
    knowledge_cutoff: str = "2025-10-01",
) -> str:
    """Load KalshiBench from HuggingFace and generate a dataset card with stats.
    
    Args:
        dataset_name: HuggingFace dataset to load
        output_path: Path to save the stats file (defaults to volume)
        knowledge_cutoff: Date to compute time horizon from (YYYY-MM-DD)
    
    Returns:
        The generated dataset card content
    """
    from collections import defaultdict
    from datasets import load_dataset
    import numpy as np
    import re
    
    print(f"📥 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    total_questions = len(dataset)
    print(f"   Loaded {total_questions:,} questions")
    
    # Compute statistics
    categories: dict[str, int] = defaultdict(int)
    category_yes: dict[str, int] = defaultdict(int)
    ground_truth_yes = 0
    ground_truth_no = 0
    close_times: list[str] = []
    question_lengths: list[int] = []
    description_lengths: list[int] = []
    all_words: set[str] = set()
    
    for item in dataset:
        # Category distribution
        cat = item.get("category", "unknown")
        categories[cat] += 1
        
        # Ground truth distribution
        gt = item.get("ground_truth", "").lower()
        if gt == "yes":
            ground_truth_yes += 1
            category_yes[cat] += 1
        elif gt == "no":
            ground_truth_no += 1
        
        # Date range
        ct = item.get("close_time", "")
        if ct:
            close_times.append(ct)
        
        # Text lengths and vocabulary
        q = item.get("question", "")
        d = item.get("description", "")
        question_lengths.append(len(q))
        description_lengths.append(len(d))
        
        # Extract words for vocabulary
        words = re.findall(r'\b\w+\b', (q + " " + d).lower())
        all_words.update(words)
    
    # Compute aggregates
    date_range_earliest = min(close_times) if close_times else "N/A"
    date_range_latest = max(close_times) if close_times else "N/A"
    avg_question_length = float(np.mean(question_lengths)) if question_lengths else 0
    median_question_length = float(np.median(question_lengths)) if question_lengths else 0
    avg_description_length = float(np.mean(description_lengths)) if description_lengths else 0
    vocabulary_size = len(all_words)
    
    # Temporal span and time horizon
    temporal_span_days = 0
    mean_time_horizon = 0.0
    if close_times:
        earliest_dt = datetime.fromisoformat(date_range_earliest.replace('Z', '+00:00').split('T')[0])
        latest_dt = datetime.fromisoformat(date_range_latest.replace('Z', '+00:00').split('T')[0])
        temporal_span_days = (latest_dt - earliest_dt).days
        
        # Time horizon from knowledge cutoff
        cutoff_dt = datetime.fromisoformat(knowledge_cutoff)
        horizons = []
        for ct in close_times:
            close_dt = datetime.fromisoformat(ct.replace('Z', '+00:00').split('T')[0])
            days_ahead = (close_dt - cutoff_dt).days
            if days_ahead > 0:
                horizons.append(days_ahead)
        mean_time_horizon = float(np.mean(horizons)) if horizons else 0.0
    
    # Sort categories by count
    sorted_categories = sorted(categories.items(), key=lambda x: -x[1])
    
    # Generate dataset card
    yes_rate = ground_truth_yes / total_questions if total_questions > 0 else 0
    no_rate = ground_truth_no / total_questions if total_questions > 0 else 0
    
    # Category table with yes rates (two-column layout)
    cat_rows = []
    half = (len(sorted_categories) + 1) // 2
    for i in range(half):
        cat1, count1 = sorted_categories[i]
        pct1 = count1 / total_questions * 100
        yes_pct1 = category_yes[cat1] / count1 * 100 if count1 > 0 else 0
        
        if i + half < len(sorted_categories):
            cat2, count2 = sorted_categories[i + half]
            pct2 = count2 / total_questions * 100
            yes_pct2 = category_yes[cat2] / count2 * 100 if count2 > 0 else 0
            cat_rows.append(f"| {cat1} | {count1} | {pct1:.1f} | {yes_pct1:.1f} | {cat2} | {count2} | {pct2:.1f} | {yes_pct2:.1f} |")
        else:
            cat_rows.append(f"| {cat1} | {count1} | {pct1:.1f} | {yes_pct1:.1f} | | | | |")
    
    category_table = "\n".join(cat_rows)
    
    card_content = f"""---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - prediction-markets
  - forecasting
  - calibration
  - benchmark
size_categories:
  - 1K<n<10K
---

# KalshiBench

A benchmark dataset for evaluating LLM forecasting calibration using real-world
prediction market data from Kalshi.

## Dataset Description

KalshiBench contains {total_questions:,} cleaned, deduplicated prediction market
questions with known outcomes. The dataset spans {len(categories)} categories with a 
{yes_rate:.0%}/{no_rate:.0%} yes/no class split, providing sufficient base rate variation 
for meaningful calibration assessment.

## Dataset Statistics

| Statistic | Value | Statistic | Value |
|-----------|-------|-----------|-------|
| Total Questions | {total_questions} | Categories | {len(categories)} |
| Temporal Span | {temporal_span_days} days | Yes Rate | {yes_rate:.1%} |
| Mean Time Horizon | {mean_time_horizon:.1f} days | No Rate | {no_rate:.1%} |
| Vocabulary Size | {vocabulary_size:,} words | Median Question Length | {median_question_length:.0f} chars |

**Date Range:** {date_range_earliest} to {date_range_latest}

## Category Distribution

| Category | N | % | Yes% | Category | N | % | Yes% |
|----------|---|---|------|----------|---|---|------|
{category_table}

## Deduplication and Quality Control

Raw prediction market data contains redundant questions (e.g., daily instances of recurring markets).
We limit to 2 questions per series ticker to preserve diversity while reducing redundancy. All questions
include detailed resolution criteria in the description field, ensuring unambiguous ground truth.

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_name}", split="train")

# Filter by date (for temporal evaluation)
cutoff = "2024-06-01"
future_questions = dataset.filter(lambda x: x["close_time"] >= cutoff)

# Access a question
q = dataset[0]
print(f"Question: {{q['question']}}")
print(f"Ground Truth: {{q['ground_truth']}}")
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (series_ticker) |
| `question` | string | The prediction market question |
| `description` | string | Detailed description/resolution criteria |
| `category` | string | Question category |
| `close_time` | string | When the market closed (ISO format) |
| `ground_truth` | string | Resolved outcome ("yes" or "no") |
| `series_ticker` | string | Original Kalshi series ticker |
| `source` | string | Data source ("kalshi") |

## Citation

```bibtex
@misc{{kalshibench2024,
  title={{KalshiBench: A Benchmark for LLM Forecasting Calibration}},
  year={{2024}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/datasets/{dataset_name}}}}}
}}
```

## License

MIT

---

*Generated: {datetime.now(UTC).isoformat()}*
"""
    
    # Save to file
    if output_path is None:
        output_path = os.path.join(DATASET_OUTPUT_DIR, "dataset_stats")
    os.makedirs(output_path, exist_ok=True)
    
    stats_file = os.path.join(output_path, "DATASET_CARD.md")
    with open(stats_file, "w") as f:
        f.write(card_content)
    print(f"💾 Saved dataset card to: {stats_file}")
    
    # Also save raw stats as JSON
    import json
    category_stats = {
        cat: {"count": count, "pct": count / total_questions * 100, "yes_pct": category_yes[cat] / count * 100 if count > 0 else 0}
        for cat, count in sorted_categories
    }
    stats_json = {
        "dataset_name": dataset_name,
        "total_questions": total_questions,
        "date_range": {"earliest": date_range_earliest, "latest": date_range_latest},
        "temporal_span_days": temporal_span_days,
        "mean_time_horizon_days": mean_time_horizon,
        "knowledge_cutoff": knowledge_cutoff,
        "ground_truth_distribution": {"yes": ground_truth_yes, "no": ground_truth_no, "yes_rate": yes_rate, "no_rate": no_rate},
        "num_categories": len(categories),
        "category_stats": category_stats,
        "vocabulary_size": vocabulary_size,
        "avg_question_length": avg_question_length,
        "median_question_length": median_question_length,
        "avg_description_length": avg_description_length,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    json_file = os.path.join(output_path, "dataset_stats.json")
    with open(json_file, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"💾 Saved raw stats to: {json_file}")
    
    _ = kalshibench_volume.commit()
    
    return card_content


@app.local_entrypoint()
def stats():
    """Generate dataset stats from HuggingFace KalshiBench."""
    result = cast(_ModalRemoteFn, generate_dataset_stats).remote()
    print("\n" + "=" * 60)
    print("Generated Dataset Card:")
    print("=" * 60)
    print(result)


@app.local_entrypoint()
def full():
    """Run full NeurIPS benchmark across all model tiers."""
    # _ = cast(_ModalRemoteFn, create_kalshibench_dataset).remote(
    #     output_path=DATASET_OUTPUT_DIR,
    # )
    _ = cast(_ModalRemoteFn, run_neurips_full).remote()

