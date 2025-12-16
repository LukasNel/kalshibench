#!/usr/bin/env python3
"""
Create KalshiBench Dataset

This script creates a cleaned, deduplicated benchmark dataset from raw prediction
market data and optionally uploads it to HuggingFace.

The resulting dataset is ready for direct use in benchmark evaluation without
any additional preprocessing.

Usage:
    # Create and save locally
    python create_kalshibench.py --output kalshibench_v1
    
    # Create and upload to HuggingFace  
    python create_kalshibench.py --upload 2084Collective/kalshibench-v2
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("create_kalshibench")


@dataclass
class DatasetStats:
    """Statistics about the created dataset."""
    raw_count: int
    after_ground_truth_filter: int
    after_dedup: int
    final_count: int
    categories: dict[str, int]
    ground_truth_distribution: dict[str, int]
    date_range: dict[str, str]
    avg_question_length: float
    avg_description_length: float


def create_kalshibench(
    source_dataset: str = "2084Collective/prediction-markets-historical-v5-cleaned",
    min_description_length: int = 10,
    require_ground_truth: bool = True,
) -> tuple[Dataset, DatasetStats]:
    """
    Create a cleaned, deduplicated KalshiBench dataset.
    
    Steps:
    1. Load raw prediction market data
    2. Filter to questions with valid ground truth
    3. Deduplicate by series_ticker (keep first occurrence)
    4. Clean and validate fields
    5. Compute statistics
    
    Args:
        source_dataset: HuggingFace dataset to load from
        min_description_length: Minimum description length to include
        require_ground_truth: Only include resolved questions
        
    Returns:
        Tuple of (cleaned Dataset, DatasetStats)
    """
    logger.info("=" * 60)
    logger.info("CREATING KALSHIBENCH DATASET")
    logger.info("=" * 60)
    
    # Step 1: Load raw data
    logger.info(f"📥 Loading source: {source_dataset}")
    raw_dataset = load_dataset(source_dataset, split="train")
    raw_count = len(raw_dataset)
    logger.info(f"   Raw dataset size: {raw_count:,}")
    
    # Step 2: Filter to valid ground truth
    logger.info("🔍 Filtering to valid ground truth...")
    valid_outcomes = {"yes", "no"}
    
    def has_valid_ground_truth(example):
        outcome = example.get("winning_outcome", "")
        if outcome:
            outcome = outcome.lower().strip()
        return outcome in valid_outcomes
    
    dataset = raw_dataset.filter(has_valid_ground_truth)
    after_gt_count = len(dataset)
    logger.info(f"   After ground truth filter: {after_gt_count:,} ({after_gt_count/raw_count:.1%})")
    
    # Step 3: Limit to 2 questions per series_ticker
    logger.info("🔄 Limiting to 2 questions per series_ticker...")
    ticker_counts: dict[str, int] = defaultdict(int)
    MAX_PER_TICKER = 2
    
    def limit_per_ticker(example):
        ticker = example.get("series_ticker", "")
        if not ticker:
            # Keep items without ticker (can't dedupe)
            return True
        if ticker_counts[ticker] >= MAX_PER_TICKER:
            return False
        ticker_counts[ticker] += 1
        return True
    
    dataset = dataset.filter(limit_per_ticker)
    after_dedup_count = len(dataset)
    logger.info(f"   After limiting: {after_dedup_count:,} ({after_dedup_count/after_gt_count:.1%})")
    logger.info(f"   Removed {after_gt_count - after_dedup_count:,} excess duplicates")
    
    # Step 4: Clean and standardize fields
    logger.info("🧹 Cleaning and standardizing fields...")
    
    def clean_example(example):
        """Clean and standardize a single example."""
        # Standardize ground truth
        outcome = example.get("winning_outcome", "").lower().strip()
        
        # Clean text fields
        question = (example.get("question") or "").strip()
        description = (example.get("description") or "").strip()
        category = (example.get("category") or "unknown").strip()
        
        # Parse close_time
        close_time = example.get("close_time") or ""
        
        # Get market probability if available
        market_prob = example.get("last_price")
        if market_prob is not None:
            try:
                market_prob = float(market_prob)
                if not (0 <= market_prob <= 1):
                    market_prob = None
            except (ValueError, TypeError):
                market_prob = None
        
        return {
            "id": example.get("series_ticker") or f"kalshi_{example.get('id', 'unknown')}",
            "question": question,
            "description": description[:5000] if description else "",  # Truncate very long descriptions
            "category": category,
            "close_time": close_time,
            "ground_truth": outcome,
            "market_probability": market_prob,
            # Preserve original fields for reference
            "series_ticker": example.get("series_ticker", ""),
            "source": "kalshi",
        }
    
    cleaned_data = []
    for example in tqdm(dataset, desc="Cleaning"):
        cleaned = clean_example(example)
        # Filter out examples with empty questions
        if cleaned["question"] and len(cleaned["description"]) >= min_description_length:
            cleaned_data.append(cleaned)
    
    final_count = len(cleaned_data)
    logger.info(f"   Final dataset size: {final_count:,}")
    
    # Step 5: Create HuggingFace Dataset
    logger.info("📦 Creating HuggingFace Dataset...")
    final_dataset = Dataset.from_list(cleaned_data)
    
    # Step 6: Compute statistics
    logger.info("📊 Computing statistics...")
    
    # Category distribution
    categories: dict[str, int] = defaultdict(int)
    for item in cleaned_data:
        categories[item["category"]] += 1
    
    # Ground truth distribution
    gt_dist = {"yes": 0, "no": 0}
    for item in cleaned_data:
        gt_dist[item["ground_truth"]] += 1
    
    # Date range
    close_times = [item["close_time"] for item in cleaned_data if item["close_time"]]
    date_range = {
        "earliest": min(close_times) if close_times else "",
        "latest": max(close_times) if close_times else "",
    }
    
    # Text lengths
    q_lengths = [len(item["question"]) for item in cleaned_data]
    d_lengths = [len(item["description"]) for item in cleaned_data]
    
    stats = DatasetStats(
        raw_count=raw_count,
        after_ground_truth_filter=after_gt_count,
        after_dedup=after_dedup_count,
        final_count=final_count,
        categories=dict(sorted(categories.items(), key=lambda x: -x[1])),
        ground_truth_distribution=gt_dist,
        date_range=date_range,
        avg_question_length=float(np.mean(q_lengths)) if q_lengths else 0,
        avg_description_length=float(np.mean(d_lengths)) if d_lengths else 0,
    )
    
    # Log summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET CREATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Final size: {stats.final_count:,} questions")
    logger.info(f"   Date range: {stats.date_range['earliest']} to {stats.date_range['latest']}")
    logger.info(f"   Ground truth: {stats.ground_truth_distribution['yes']} yes, {stats.ground_truth_distribution['no']} no")
    logger.info(f"   Categories: {len(stats.categories)}")
    logger.info(f"   Top categories:")
    for cat, count in list(stats.categories.items())[:5]:
        logger.info(f"      {cat}: {count}")
    
    return final_dataset, stats


def save_dataset(
    dataset: Dataset,
    stats: DatasetStats,
    output_path: str,
) -> None:
    """Save dataset and stats locally."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_path = output_dir / "dataset"
    logger.info(f"💾 Saving dataset to {dataset_path}")
    dataset.save_to_disk(str(dataset_path))
    
    # Save stats
    stats_path = output_dir / "stats.json"
    logger.info(f"💾 Saving stats to {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    
    # Save README
    readme_path = output_dir / "README.md"
    readme_content = f"""# KalshiBench Dataset

A cleaned, deduplicated benchmark dataset for evaluating LLM forecasting calibration
using Kalshi prediction market data.

## Statistics

- **Total Questions**: {stats.final_count:,}
- **Date Range**: {stats.date_range['earliest']} to {stats.date_range['latest']}
- **Ground Truth Distribution**: {stats.ground_truth_distribution['yes']} yes, {stats.ground_truth_distribution['no']} no
- **Categories**: {len(stats.categories)}

## Processing Steps

1. Loaded {stats.raw_count:,} raw examples from source
2. Filtered to {stats.after_ground_truth_filter:,} with valid ground truth (yes/no)
3. Limited to 2 questions per series_ticker: {stats.after_dedup:,} questions
4. Cleaned and validated to {stats.final_count:,} final questions

## Fields

- `id`: Unique identifier (series_ticker)
- `question`: The prediction market question
- `description`: Detailed description/resolution criteria
- `category`: Question category (e.g., "Politics", "Economics")
- `close_time`: When the market closed (ISO format)
- `ground_truth`: Resolved outcome ("yes" or "no")
- `series_ticker`: Original Kalshi series ticker
- `source`: Data source ("kalshi")

## Usage

```python
from datasets import load_from_disk

dataset = load_from_disk("{output_path}/dataset")
print(f"Loaded {{len(dataset)}} questions")
```

## Created

{datetime.now().isoformat()}
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    logger.info(f"✅ Dataset saved to {output_dir}")


def upload_dataset(
    dataset: Dataset,
    stats: DatasetStats,
    repo_id: str,
    private: bool = False,
) -> None:
    """Upload dataset to HuggingFace Hub."""
    logger.info(f"📤 Uploading to HuggingFace: {repo_id}")
    
    # Create dataset card
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

KalshiBench contains {stats.final_count:,} cleaned, deduplicated prediction market
questions with known outcomes. Each question includes the market's final probability,
enabling comparison between LLM predictions and crowd wisdom.

### Statistics

| Metric | Value |
|--------|-------|
| Total Questions | {stats.final_count:,} |
| Date Range | {stats.date_range['earliest']} to {stats.date_range['latest']} |
| Yes Outcomes | {stats.ground_truth_distribution['yes']} ({stats.ground_truth_distribution['yes']/stats.final_count:.1%}) |
| No Outcomes | {stats.ground_truth_distribution['no']} ({stats.ground_truth_distribution['no']/stats.final_count:.1%}) |
| Categories | {len(stats.categories)} |
| Avg Question Length | {stats.avg_question_length:.0f} chars |
| Avg Description Length | {stats.avg_description_length:.0f} chars |

### Top Categories

| Category | Count |
|----------|-------|
{"".join(f"| {cat} | {count} |" + chr(10) for cat, count in list(stats.categories.items())[:10])}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}", split="train")

# Filter by date (for temporal evaluation)
cutoff = "2024-06-01"
future_questions = dataset.filter(lambda x: x["close_time"] >= cutoff)

# Access a question
q = dataset[0]
print(f"Question: {{q['question']}}")
print(f"Ground Truth: {{q['ground_truth']}}")
print(f"Market Probability: {{q['market_probability']}}")
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
| `market_probability` | float | Final market price (0-1), null if unavailable |
| `series_ticker` | string | Original Kalshi series ticker |
| `source` | string | Data source ("kalshi") |

## Citation

```bibtex
@misc{{kalshibench2024,
  title={{KalshiBench: A Benchmark for LLM Forecasting Calibration}},
  year={{2024}},
  publisher={{HuggingFace}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## License

MIT
"""
    
    # Push to hub
    dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message=f"Upload KalshiBench v1 ({stats.final_count:,} questions)",
    )
    
    # Note: Dataset card is auto-generated, but you can update it manually on HuggingFace
    logger.info(f"✅ Uploaded to https://huggingface.co/datasets/{repo_id}")
    logger.info("   Note: Update the dataset card on HuggingFace for full documentation")


def main():
    parser = argparse.ArgumentParser(
        description="Create KalshiBench dataset from raw prediction market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="2084Collective/prediction-markets-historical-v5-cleaned",
        help="Source HuggingFace dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Local output directory (e.g., 'kalshibench_v1')",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="HuggingFace repo ID to upload to (e.g., '2084Collective/kalshibench-v2')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make uploaded dataset private",
    )
    parser.add_argument(
        "--min-description-length",
        type=int,
        default=10,
        help="Minimum description length to include (default: 10)",
    )
    
    args = parser.parse_args()
    
    if not args.output and not args.upload:
        parser.error("Must specify --output and/or --upload")
    
    # Create dataset
    dataset, stats = create_kalshibench(
        source_dataset=args.source,
        min_description_length=args.min_description_length,
    )
    
    # Save locally
    if args.output:
        save_dataset(dataset, stats, args.output)
    
    # Upload to HuggingFace
    if args.upload:
        upload_dataset(dataset, stats, args.upload, private=args.private)
    
    logger.info("")
    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()

