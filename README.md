# KalshiBench

**A benchmark for evaluating LLM forecasting calibration using real-world prediction markets.**

KalshiBench tests whether language models can make well-calibrated probabilistic predictions on questions with verifiable real-world outcomes. Unlike traditional benchmarks that measure accuracy on static knowledge, KalshiBench evaluates models on temporally-filtered prediction market questions—ensuring models cannot have memorized the answers during training.

---

## Table of Contents

- [Why KalshiBench?](#why-kalshibench)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Metrics Explained](#metrics-explained)
- [Available Models](#available-models)
- [Output Format](#output-format)
- [Running for Research](#running-for-research)
- [Citation](#citation)

---

## Why KalshiBench?

### The Problem with Existing Benchmarks

Most LLM benchmarks evaluate **accuracy**—whether a model gets the right answer. But for many real-world applications, we care about **calibration**—whether a model's confidence matches its actual accuracy.

A model that says "I'm 90% confident" should be correct 90% of the time on similar questions. Poor calibration leads to:
- Overconfident wrong answers that users trust
- Underconfident correct answers that users ignore
- Inability to meaningfully aggregate model predictions

### Why Prediction Markets?

Prediction markets offer unique advantages for LLM evaluation:

| Property | Traditional Benchmarks | KalshiBench |
|----------|----------------------|-------------|
| Ground truth | Human annotations | Real-world outcomes |
| Temporal validity | Static (can be memorized) | Post-training-cutoff |
| Calibration signal | Binary (correct/wrong) | Continuous probabilities |
| Domain | Academic/synthetic | Real-world forecasting |

### Research Questions

KalshiBench enables research into:

1. **Epistemic Calibration**: Can LLMs learn to "know what they don't know"?
2. **Reasoning vs. Calibration**: Do reasoning models (o1, DeepSeek-R1) calibrate better?
3. **Scaling Effects**: Does model size improve calibration, or just accuracy?
4. **Provider Differences**: How do different model families compare on forecasting?

---

## Key Features

- **Temporal Filtering**: Automatically uses the latest knowledge cutoff among selected models
- **20+ Models**: GPT-5.2, Claude Opus 4.5, o1, Gemini, Llama, Qwen, DeepSeek, and more
- **Comprehensive Metrics**: Brier Score, ECE, calibration curves, overconfidence rates
- **Paper-Ready Output**: JSON results + prompts for generating Methods/Results sections
- **Async Evaluation**: Parallel API calls with configurable concurrency
- **Cost Efficient**: Budget model options for development, flagship models for publication

---

## Installation

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for model providers

### Setup with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-org/kalshibench.git
cd kalshibench

# Install dependencies (creates virtual environment automatically)
uv sync

# Or install with optional dependencies
uv sync --extra training  # For GRPO training
uv sync --extra all       # Everything
```

### Setup with pip

```bash
# Clone the repository
git clone https://github.com/your-org/kalshibench.git
cd kalshibench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# Or with optional dependencies
pip install -e ".[training]"  # For GRPO training
pip install -e ".[all]"       # Everything
```

### API Keys

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
GEMINI_API_KEY=...
```

### Running with uv

```bash
# uv automatically uses the project's virtual environment
uv run python kalshibench.py --models gpt-4o-mini --samples 10

# Or activate the environment manually
source .venv/bin/activate
python kalshibench.py --models gpt-4o-mini --samples 10
```

---

## Quick Start

### Test Your Setup

```bash
# Load environment and run quick test
source .env  # or use run_test.sh
python kalshibench.py --models gpt-4o-mini --samples 10
```

### Basic Evaluation

```bash
# Compare a few models
python kalshibench.py --models gpt-4o claude-sonnet-4.5 --samples 100

# List all available models
python kalshibench.py --list-models
```

### Full Research Run

```bash
# Run comprehensive benchmark for paper
python run_neurips_benchmark.py --full
```

---

## How It Works

### 1. Data Source & Dataset Creation

KalshiBench uses prediction market data from [Kalshi](https://kalshi.com), a CFTC-regulated prediction market platform. The dataset is created using an extensible extraction pipeline that can pull from multiple prediction market platforms.

#### Dataset Creation Architecture

The dataset creation system (`dataset_creator.py`) uses an abstract factory pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                PredictionMarketDataOrchestrator                 │
│  - Coordinates multiple extractors                              │
│  - Creates unified schema across platforms                      │
│  - Pushes to HuggingFace Hub                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ KalshiExtractor │  │PolymarketExtr. │  │ Future Platform │
│                 │  │                 │  │                 │
│ - Kalshi API    │  │ - Gamma API     │  │ - Your API      │
│ - Binary yes/no │  │ - Multi-outcome │  │ - Custom schema │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

#### Abstract Base Class

All extractors inherit from `PredictionMarketDataExtractor`:

```python
class PredictionMarketDataExtractor(ABC):
    @abstractmethod
    def get_market_name(self) -> str:
        """Return platform name (e.g., 'kalshi', 'polymarket')"""
        
    @abstractmethod
    def fetch_closed_markets(self, max_markets: int) -> list[dict]:
        """Fetch resolved markets from the platform API"""
        
    @abstractmethod
    def get_dataset_features(self) -> Features:
        """Return HuggingFace Features schema"""
        
    @abstractmethod
    def to_unified_format(self, market: dict) -> dict:
        """Convert platform-specific data to unified schema"""
```

#### Kalshi Data Extraction

The `KalshiDataExtractor` fetches from Kalshi's public API:

```python
# API endpoint for settled events with nested markets
url = "https://api.elections.kalshi.com/trade-api/v2/events"
params = {
    "status": "settled",      # Only resolved markets
    "limit": 200,             # Max per page
    "with_nested_markets": "true"  # Include market details
}
```

**Data fields extracted from Kalshi:**

| Field | Description |
|-------|-------------|
| `market_id` | Unique market ticker |
| `question` | Market title/question |
| `description` | Resolution rules (primary + secondary) |
| `category` | Topic category from parent event |
| `winning_outcome` | Result: "yes" or "no" |
| `close_time` | When the market resolved |
| `volume` | Total trading volume |
| `last_price` | Final market price (0-100 cents) |
| `series_ticker` | Group identifier for related markets |

#### Polymarket Data Extraction

The `PolymarketDataExtractor` fetches from Polymarket's Gamma API:

```python
url = "https://gamma-api.polymarket.com/markets"
params = {
    "closed": "true",  # Only resolved markets
    "limit": 100,
    "offset": offset
}
```

**Key difference:** Polymarket supports multi-outcome markets (not just yes/no), so outcome parsing is more complex:

```python
def parse_winning_outcome(outcomes_str, outcome_prices_str):
    """Determine winner by finding outcome with highest final price"""
    outcomes = json.loads(outcomes_str)   # ["Yes", "No"] or ["Biden", "Trump", "Other"]
    prices = json.loads(outcome_prices_str)  # [0.65, 0.35] or [0.4, 0.55, 0.05]
    
    max_idx = prices.index(max(prices))
    return outcomes[max_idx]  # Winner is highest-priced outcome
```

#### Unified Schema

The orchestrator normalizes platform-specific data into a unified schema:

```python
unified_schema = Features({
    "platform": Value("string"),        # Source platform
    "market_id": Value("string"),       # Unique identifier
    "question": Value("string"),        # The prediction question
    "description": Value("string"),     # Resolution criteria
    "category": Value("string"),        # Topic category
    "winning_outcome": Value("string"), # Actual result
    "outcomes": Value("string"),        # JSON: all possible outcomes
    "outcome_prices": Value("string"),  # JSON: final prices
    "close_time": Value("string"),      # Resolution timestamp
    "volume": Value("float64"),         # Trading volume
    "liquidity": Value("float64"),      # Market liquidity
    "open_interest": Value("float64"),  # Open positions
    "original": Value("string"),        # Raw API response (for debugging)
    # ... additional fields
})
```

#### Running Dataset Creation

```bash
# Create fresh dataset from APIs
python dataset_creator.py

# This will:
# 1. Fetch all settled markets from Kalshi
# 2. Convert to unified schema
# 3. Save locally as parquet
# 4. Push to HuggingFace Hub
```

#### Data Cleaning

After fetching, the dataset is cleaned to ensure quality:

```python
def clean_dataset(data_item):
    """Filter to only Kalshi binary markets with clear outcomes"""
    accept = True
    accept = accept and data_item['platform'] == 'kalshi'
    accept = accept and data_item['winning_outcome'].lower() in ['yes', 'no']
    return accept

# Result: 2084Collective/prediction-markets-historical-v5-cleaned
```

#### Adding New Platforms

To add a new prediction market platform:

```python
class NewPlatformExtractor(PredictionMarketDataExtractor):
    def get_market_name(self) -> str:
        return "new_platform"
    
    def fetch_closed_markets(self, max_markets: int) -> list[dict]:
        # Implement API fetching
        pass
    
    def get_dataset_features(self) -> Features:
        # Define platform-specific schema
        pass
    
    def to_unified_format(self, market: dict) -> dict:
        # Map to unified schema
        pass

# Add to orchestrator
orchestrator = PredictionMarketDataOrchestrator([
    KalshiDataExtractor(),
    PolymarketDataExtractor(),
    NewPlatformExtractor(),  # Your new extractor
])
```

#### HuggingFace Dataset

The processed dataset is available on HuggingFace:

```python
from datasets import load_dataset

# Load the cleaned Kalshi dataset
dataset = load_dataset("2084Collective/prediction-markets-historical-v5-cleaned")

# Inspect
print(f"Total markets: {len(dataset['train'])}")
print(dataset['train'][0])
```

**Dataset URL:** [huggingface.co/datasets/2084Collective/prediction-markets-historical-v5-cleaned](https://huggingface.co/datasets/2084Collective/prediction-markets-historical-v5-cleaned)

---

### 2. Data Fields

The dataset contains:

- **Question**: The prediction market question (e.g., "Will inflation exceed 3% in Q4 2025?")
- **Description**: Context and resolution criteria
- **Close Time**: When the market resolved
- **Winning Outcome**: The actual result (yes/no)
- **Category**: Topic area (politics, economics, sports, etc.)

### 2. Temporal Filtering

To ensure fair evaluation, KalshiBench only uses questions that resolved **after** the latest knowledge cutoff among all models being tested.

```
Example: Testing GPT-5.2 (cutoff: 2025-10-01) and Claude Opus 4.5 (cutoff: 2025-04-01)

→ Benchmark uses cutoff: 2025-10-01 (the latest)
→ Only questions resolving after October 2025 are included
→ Neither model could have seen the outcomes during training
```

This is automatically computed:

```python
# From kalshibench.py
def get_max_knowledge_cutoff(model_keys: list[str]) -> str:
    """Get the latest knowledge cutoff date from selected models."""
    cutoffs = [MODELS[key].knowledge_cutoff for key in model_keys if key in MODELS]
    return max(cutoffs)  # Latest date ensures fairness
```

### 3. Evaluation Protocol

For each question, the model receives:

```
System: You are an expert forecaster evaluating prediction market questions...

User: Question: {question}
      Description: {description}
      
      Based on the information provided, predict whether this will resolve 
      to "yes" or "no".
```

The model must respond with:

```xml
<think>
[Reasoning about the prediction, considering base rates, relevant factors, 
and uncertainty]
</think>
<answer>[yes or no]</answer>
<confidence>[0-100, representing P(yes)]</confidence>
```

### 4. Metric Computation

The benchmark computes:

1. **Classification metrics** from the binary yes/no predictions
2. **Calibration metrics** from the confidence scores vs. actual outcomes
3. **Per-category breakdowns** for fine-grained analysis
4. **Reliability diagrams** for visualization

---

## Metrics Explained

### Classification Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | % of correct predictions | Higher is better |
| **Macro F1** | Average of F1 for yes/no classes | Handles class imbalance |
| **Precision/Recall** | Per-class performance | Identifies prediction biases |

### Calibration Metrics (Primary Focus)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Brier Score** | Mean((confidence - outcome)²) | Lower is better. 0 = perfect, 1 = worst |
| **Brier Skill Score** | 1 - (Brier / Brier_climatology) | Improvement over always-predict-base-rate |
| **ECE** | Σ (bin_size/N) × \|confidence - accuracy\| | Expected Calibration Error. Lower = better calibrated |
| **MCE** | max(\|confidence - accuracy\|) per bin | Maximum Calibration Error. Worst-case bin |
| **ACE** | ECE with equal-mass bins | Adaptive CE. Less sensitive to bin boundaries |
| **Log Loss** | -Σ [y·log(p) + (1-y)·log(1-p)] | Penalizes confident wrong predictions heavily |

### Overconfidence Metrics

| Metric | Description |
|--------|-------------|
| **Overconfidence Rate @X%** | % of predictions with confidence >X% that are wrong |
| **Avg Confidence When Wrong** | Mean confidence on incorrect predictions |
| **Avg Confidence When Right** | Mean confidence on correct predictions |

### Understanding Calibration

A well-calibrated model satisfies:

```
P(correct | confidence = p) ≈ p
```

For example, among all predictions where the model said "70% confident", approximately 70% should be correct.

**Reliability Diagram**: Plots predicted confidence (x-axis) vs. actual accuracy (y-axis). A perfectly calibrated model follows the diagonal line.

```
Actual
Accuracy
  1.0 |        ╱ (perfect calibration)
      |      ╱
  0.5 |    ╱  
      |  ╱
  0.0 |╱___________
      0   0.5   1.0
        Confidence
```

---

## Available Models

### OpenAI

| Key | Model | Cutoff | Notes |
|-----|-------|--------|-------|
| `gpt-5.2` | GPT-5.2 | 2025-10-01 | Latest flagship |
| `gpt-5.1` | GPT-5.1 | 2025-08-01 | |
| `gpt-4o` | GPT-4o | 2024-10-01 | Multimodal flagship |
| `gpt-4o-mini` | GPT-4o-mini | 2024-10-01 | Fast, affordable |
| `o1` | o1 | 2024-10-01 | Reasoning model |
| `o1-mini` | o1-mini | 2024-10-01 | Faster reasoning |
| `o3-mini` | o3-mini | 2025-01-01 | Latest reasoning |

### Anthropic

| Key | Model | Cutoff | Notes |
|-----|-------|--------|-------|
| `claude-opus-4.5` | Claude Opus 4.5 | 2025-04-01 | Most capable |
| `claude-sonnet-4.5` | Claude Sonnet 4.5 | 2025-04-01 | Balanced |
| `claude-3-5-sonnet` | Claude 3.5 Sonnet | 2024-04-01 | Previous gen |
| `claude-3-5-haiku` | Claude 3.5 Haiku | 2024-04-01 | Fast |

### Together AI (Open-source)

| Key | Model | Cutoff | Notes |
|-----|-------|--------|-------|
| `qwen-2.5-72b` | Qwen 2.5 72B | 2024-09-01 | |
| `qwen-qwq-32b` | Qwen QwQ 32B | 2024-09-01 | Reasoning model |
| `llama-3.3-70b` | Llama 3.3 70B | 2024-12-01 | |
| `llama-3.1-405b` | Llama 3.1 405B | 2024-03-01 | Largest open |
| `deepseek-v3` | DeepSeek V3 | 2024-11-01 | |
| `deepseek-r1` | DeepSeek R1 | 2024-11-01 | Reasoning model |
| `mistral-large` | Mistral Large | 2024-07-01 | |

### Google

| Key | Model | Cutoff | Notes |
|-----|-------|--------|-------|
| `gemini-2.0-flash` | Gemini 2.0 Flash | 2024-08-01 | Fast |
| `gemini-1.5-pro` | Gemini 1.5 Pro | 2024-04-01 | |

---

## Output Format

### Directory Structure

```
kalshibench_results/
├── summary_TIMESTAMP.json         # Aggregated results
├── report_TIMESTAMP.md            # Human-readable report
├── metadata_TIMESTAMP.json        # Benchmark configuration
├── paper_methods_prompt_*.txt     # LLM prompt for Methods section
├── paper_results_prompt_*.txt     # LLM prompt for Results section
└── MODEL_NAME_TIMESTAMP.json      # Per-model detailed results
```

### Summary JSON Schema

```json
{
  "benchmark": "KalshiBench",
  "timestamp": "2025-12-15T10:30:00",
  "knowledge_cutoff_used": "2025-10-01",
  "num_models": 4,
  "num_samples": 200,
  "models": {
    "GPT-5.2": {
      "knowledge_cutoff": "2025-10-01",
      "accuracy": 0.73,
      "macro_f1": 0.71,
      "brier_score": 0.198,
      "ece": 0.082,
      "overconfidence_rate_80": 0.23
    }
  },
  "leaderboard": {
    "by_accuracy": [...],
    "by_brier_score": [...],
    "by_calibration": [...]
  }
}
```

### Per-Model JSON Schema

```json
{
  "model_name": "GPT-5.2",
  "model_config": {...},
  "timestamp": "2025-12-15T10:30:00",
  "num_samples": 200,
  
  "accuracy": 0.73,
  "precision_yes": 0.75,
  "recall_yes": 0.70,
  "f1_yes": 0.72,
  "macro_f1": 0.71,
  
  "brier_score": 0.198,
  "brier_skill_score": 0.21,
  "ece": 0.082,
  "mce": 0.15,
  "log_loss": 0.58,
  
  "avg_confidence": 0.72,
  "overconfidence_rate_80": 0.23,
  
  "reliability_diagram": [
    {"bin": "0.0-0.1", "avg_confidence": 0.05, "avg_accuracy": 0.08, "count": 12},
    {"bin": "0.1-0.2", "avg_confidence": 0.15, "avg_accuracy": 0.18, "count": 18},
    ...
  ],
  
  "per_category": {
    "Politics": {"accuracy": 0.68, "brier_score": 0.22, "count": 45},
    "Economics": {"accuracy": 0.76, "brier_score": 0.18, "count": 52}
  },
  
  "predictions": [...]  # Raw predictions for further analysis
}
```

---

## Running for Research

### NeurIPS Benchmark Script

For a comprehensive evaluation suitable for publication:

```bash
# Full run across all model tiers (~$70, 2-4 hours)
python run_neurips_benchmark.py --full

# Or run individual tiers
python run_neurips_benchmark.py --tier tier1_flagships   # ~$30
python run_neurips_benchmark.py --tier tier3_reasoning   # ~$25
```

### Model Tiers

| Tier | Purpose | Models |
|------|---------|--------|
| **tier1_flagships** | Best-in-class comparison | GPT-5.2, Claude Opus 4.5, Gemini, Llama 405B |
| **tier2_midtier** | Production models | GPT-4o, Claude Sonnet, Qwen 72B, DeepSeek V3 |
| **tier3_reasoning** | Reasoning analysis | o1, o1-mini, o3-mini, QwQ, DeepSeek-R1 |
| **tier4_budget** | Scaling analysis | GPT-4o-mini, Claude Haiku, small models |

### Generating Paper Sections

After running the benchmark, use the generated prompts:

```bash
# Feed to an LLM to generate paper sections
cat results/paper_methods_prompt_*.txt | pbcopy
# Paste into Claude/GPT to get Methods section

cat results/paper_results_prompt_*.txt | pbcopy
# Paste into Claude/GPT to get Results section
```

---

## Advanced Usage

### Custom Knowledge Cutoff

Override automatic cutoff computation:

```bash
python kalshibench.py --models gpt-4o claude-3-5-sonnet --cutoff 2024-06-01
```

### Adjust Concurrency

For rate-limited APIs:

```bash
python kalshibench.py --models o1 --concurrent 3  # Lower for reasoning models
```

### Programmatic Usage

```python
import asyncio
from kalshibench import KalshiBenchRunner, BenchmarkConfig, MODELS

config = BenchmarkConfig(
    num_samples=100,
    output_dir="my_results",
)

runner = KalshiBenchRunner(config)

async def main():
    results = await runner.run(["gpt-4o", "claude-sonnet-4.5"])
    
    for name, result in results.items():
        print(f"{name}: Accuracy={result.accuracy:.2%}, Brier={result.brier_score:.4f}")

asyncio.run(main())
```

---

## Limitations

1. **Temporal Scope**: Limited to questions resolving after model cutoff dates
2. **Binary Outcomes**: Currently only yes/no markets (no continuous or multi-outcome)
3. **Market Selection**: Kalshi markets may not be representative of all forecasting domains
4. **Confidence Elicitation**: Self-reported confidence may differ from true uncertainty

---

## Citation

If you use KalshiBench in your research, please cite:

```bibtex
@misc{kalshibench2025,
  title={KalshiBench: Evaluating LLM Forecasting Calibration via Prediction Markets},
  author={...},
  year={2025},
  howpublished={\url{https://github.com/your-org/kalshibench}}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please open an issue or PR for:
- Additional model support
- New metrics
- Bug fixes
- Documentation improvements

