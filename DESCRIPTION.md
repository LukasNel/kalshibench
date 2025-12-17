# KalshiBench: Evaluating LLM Forecasting Calibration via Prediction Markets

## Overview

KalshiBench is a benchmark for evaluating language model forecasting ability and **calibration** using real-world prediction market data from [Kalshi](https://kalshi.com), a CFTC-regulated prediction market platform. Unlike traditional benchmarks that measure accuracy on static knowledge, KalshiBench evaluates whether models can make **well-calibrated probabilistic predictions** on questions with verifiable real-world outcomes.

### Key Innovation

By using **temporally-filtered** prediction market questions, we ensure models cannot have memorized the answers during training—all questions resolve **after** the models' knowledge cutoffs. This provides a clean evaluation of genuine forecasting ability rather than memorization.

---

## Architecture

The KalshiBench system consists of four main components:

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  dataset_creator.py │ ───▶ │  dataset_cleaner.py │ ───▶ │create_kalshibench.py│
│                     │      │                     │      │                     │
│  Fetches raw data   │      │  Filters to valid   │      │  Deduplicates and   │
│  from Kalshi API    │      │  yes/no outcomes    │      │  creates benchmark  │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
                                                                    │
                                                                    ▼
                                                          ┌─────────────────────┐
                                                          │   kalshibench.py    │
                                                          │                     │
                                                          │  Evaluates models   │
                                                          │  and computes       │
                                                          │  calibration metrics│
                                                          └─────────────────────┘
```

---

## Dataset Creation Pipeline

### Step 1: Raw Data Collection (`dataset_creator.py`)

The `KalshiDataExtractor` class fetches closed/settled markets from the Kalshi API:

```python
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
```

**What it collects:**
- Market questions and descriptions
- Resolution rules (primary and secondary)
- Winning outcomes (yes/no)
- Market metadata (category, close time, volume, liquidity)
- Price information (last price, bid/ask spreads)
- Event-level information (series ticker, event title)

**Key features:**
- Fetches events with nested markets (`with_nested_markets=true`)
- Paginates through all settled markets using cursor-based pagination
- Preserves original JSON responses for future feature recovery
- Outputs to HuggingFace dataset format

### Step 2: Initial Cleaning (`dataset_cleaner.py`)

Filters the raw dataset to include only:
- Platform: `kalshi` (filters out any non-Kalshi data if present)
- Outcome: Only binary yes/no markets (excludes multi-outcome markets)

```python
def clean_dataset(x):
    accept = x['platform'] == 'kalshi'
    accept = accept and x['winning_outcome'].lower() in ['yes', 'no']
    return accept
```

### Step 3: Benchmark Dataset Creation (`create_kalshibench.py`)

Creates the final KalshiBench dataset with additional processing:

**Processing Steps:**

1. **Ground Truth Filter**: Only includes questions with valid `yes` or `no` outcomes
2. **Deduplication**: Limits to **2 questions per series_ticker** to avoid related questions dominating the benchmark
3. **Field Standardization**: 
   - Normalizes ground truth to lowercase
   - Truncates very long descriptions (max 5000 chars)
   - Extracts market probability from `last_price`
   - Creates unique IDs from series_ticker

**Output Schema:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (series_ticker) |
| `question` | string | The prediction market question |
| `description` | string | Resolution criteria and details |
| `category` | string | Topic category (Politics, Economics, etc.) |
| `close_time` | string | When the market closed (ISO format) |
| `ground_truth` | string | Resolved outcome ("yes" or "no") |
| `market_probability` | float | Final market price (0-1) |
| `series_ticker` | string | Original Kalshi series ticker |
| `source` | string | Data source ("kalshi") |

---

## Benchmark Evaluation (`kalshibench.py`)

### Temporal Filtering

The benchmark automatically computes the **latest knowledge cutoff** among all evaluated models, ensuring fair evaluation:

```python
def get_max_knowledge_cutoff(model_keys: list[str]) -> str:
    """Returns the latest cutoff among all models."""
    cutoffs = [MODELS[key].knowledge_cutoff for key in model_keys if key in MODELS]
    return max(cutoffs)  # Use latest date
```

Only questions that **resolve after** this cutoff are included, preventing models from having seen the outcomes during training.

### Evaluation Protocol

1. **Input**: Each model receives the question text and description
2. **Output Format**: Models must respond with XML tags:
   ```xml
   <think>
   [Reasoning about the prediction]
   </think>
   <answer>[yes or no]</answer>
   <confidence>[0-100]</confidence>
   ```
3. **Parsing**: Extracts binary prediction and confidence score
4. **Confidence Conversion**: If answer is "no", confidence is converted to P(yes) = 1 - confidence

### System Prompt

```
You are an expert forecaster evaluating prediction market questions. 
Given a question and its description, predict whether the outcome will be "yes" or "no".

Be calibrated: if you're 70% confident, you should be correct about 70% of the time 
on similar questions.
```

---

## Metrics

KalshiBench evaluates models on three categories of metrics:

### 1. Classification Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correct predictions |
| **Precision (Yes/No)** | True positives / (True positives + False positives) |
| **Recall (Yes/No)** | True positives / (True positives + False negatives) |
| **F1 Score (Yes/No)** | Harmonic mean of precision and recall |
| **Macro F1** | Average of F1 scores for both classes |

### 2. Calibration Metrics (Primary Focus)

These are the **core metrics** of KalshiBench, measuring how well a model's confidence matches its actual accuracy:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Brier Score** | $\frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2$ | Mean squared error of probability predictions. **Lower is better** (0 = perfect). |
| **Brier Skill Score** | $1 - \frac{BS}{BS_{climatology}}$ | Improvement over always predicting the base rate. **Higher is better**. |
| **ECE** (Expected Calibration Error) | $\sum_{b=1}^{B}\frac{n_b}{N}\|acc(b) - conf(b)\|$ | Weighted average of calibration error per bin. **Lower is better**. |
| **MCE** (Maximum Calibration Error) | $\max_b \|acc(b) - conf(b)\|$ | Worst-calibrated confidence bin. **Lower is better**. |
| **ACE** (Adaptive Calibration Error) | ECE with equal-mass bins | More robust to uneven confidence distributions. **Lower is better**. |
| **Log Loss** | $-\frac{1}{N}\sum_{i=1}^{N}[o_i\log(p_i) + (1-o_i)\log(1-p_i)]$ | Cross-entropy loss. Heavily penalizes confident wrong predictions. **Lower is better**. |

**Why Calibration Matters:**

A model is well-calibrated if:
- When it says 70% confidence, it's correct ~70% of the time
- When it says 90% confidence, it's correct ~90% of the time

A model can be accurate but poorly calibrated (e.g., always saying 99% confidence but being correct 75% of the time).

### 3. Confidence Analysis

| Metric | Description |
|--------|-------------|
| **Average Confidence** | Mean confidence level (distance from 50%) |
| **Avg Confidence When Correct** | Mean confidence on correct predictions |
| **Avg Confidence When Wrong** | Mean confidence on incorrect predictions |
| **Overconfidence Rate @70%** | Fraction wrong among predictions with >70% confidence |
| **Overconfidence Rate @80%** | Fraction wrong among predictions with >80% confidence |
| **Overconfidence Rate @90%** | Fraction wrong among predictions with >90% confidence |

### Reliability Diagram

The benchmark generates reliability diagrams showing calibration across confidence bins:

```
| Bin       | Avg Confidence | Avg Accuracy | Count | Gap    |
|-----------|----------------|--------------|-------|--------|
| 0.0-0.1   | 0.082          | 0.120        | 25    | -0.038 |
| 0.5-0.6   | 0.553          | 0.542        | 48    | +0.011 |
| 0.9-1.0   | 0.934          | 0.856        | 67    | +0.078 |
```

- **Gap > 0**: Model is overconfident
- **Gap < 0**: Model is underconfident
- **Gap ≈ 0**: Well-calibrated

---

## Supported Models

KalshiBench supports evaluation of models from multiple providers via LiteLLM:

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-5.2, GPT-5.1, GPT-4o, GPT-4o-mini, o1, o3-mini |
| **Anthropic** | Claude Opus 4.5, Claude Sonnet 4.5, Claude 3.5 Sonnet/Haiku |
| **Google** | Gemini 3 Pro/Flash, Gemini 2.5 Pro, Gemini 2.0 Flash |
| **Together AI** | Qwen3-235B, Qwen-2.5-72B, Llama-3.3-70B, Llama-3.1-405B |
| **Fireworks AI** | DeepSeek-V3/R1, Kimi-K2 |

Each model has a configured knowledge cutoff date used for temporal filtering.

---

## Usage

### Creating the Benchmark Dataset

```bash
# Create locally
python create_kalshibench.py --output kalshibench_v1

# Upload to HuggingFace
python create_kalshibench.py --upload 2084Collective/kalshibench-v2
```

### Running Evaluations

```bash
# Evaluate models
python kalshibench.py --models gpt-4o claude-3-5-sonnet --samples 200

# List available models
python kalshibench.py --list-models

# Resume with existing results
python kalshibench.py --models gpt-4o gpt-5.2 --load-existing
```

### Output Files

The benchmark generates:

| File | Description |
|------|-------------|
| `<model>_*.json` | Individual model results with all predictions |
| `summary_*.json` | Aggregated comparison across models |
| `metadata_*.json` | Benchmark configuration and dataset statistics |
| `dataset_analysis_*.json` | Comprehensive dataset analysis |
| `report_*.md` | Human-readable markdown report |

---

## Dataset Analysis

The benchmark performs comprehensive dataset analysis including:

- **Temporal Distribution**: Date range, monthly breakdown, time horizons
- **Ground Truth Distribution**: Yes/no rates overall and by category
- **Category Analysis**: Distribution, entropy, diversity metrics
- **Question Complexity**: Length statistics, vocabulary size
- **Market Baseline**: Calibration metrics for market prices as a predictor

---

## Why Prediction Markets?

Prediction markets provide ideal ground truth for forecasting evaluation because:

1. **Real Stakes**: Questions involve real monetary stakes, ensuring market efficiency
2. **Objective Resolution**: Clear resolution criteria with verifiable outcomes
3. **Diverse Topics**: Covers politics, economics, science, sports, weather, etc.
4. **Temporal Validity**: Can enforce clean train/test splits based on resolution dates
5. **Calibrated Baseline**: Market prices themselves provide a calibrated baseline predictor

---

## Citation

```bibtex
@misc{kalshibench2024,
  title={KalshiBench: Evaluating LLM Forecasting Calibration via Prediction Markets},
  author={2084 Collective},
  year={2024},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/datasets/2084Collective/kalshibench-v2}}
}
```

---

## License

MIT

