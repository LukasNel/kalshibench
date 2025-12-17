"""
KalshiBench: Evaluating LLM Forecasting Calibration via Prediction Markets

A benchmark for evaluating language model forecasting ability and calibration
using temporally-filtered Kalshi prediction market data.

Usage:
    python kalshibench.py --models gpt-4o claude-3-5-sonnet --samples 200
"""

import os
import re
import json
import asyncio
import argparse
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal
from collections import defaultdict
from enum import Enum
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field
from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import litellm
from litellm import acompletion

# Silence litellm logging
litellm.set_verbose = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("kalshibench")


# ==============================================================================
# Configuration
# ==============================================================================

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    GOOGLE = "google"
    FIREWORKS = "fireworks"


class ModelConfig(PydanticBaseModel):
    """Configuration for a model to evaluate."""
    name: str
    litellm_model: str
    provider: ModelProvider
    knowledge_cutoff: str  # YYYY-MM-DD format
    description: str = ""
    max_tokens: int = 32768
    temperature: float = 0.7
    # Pricing per 1M tokens (input, output) in USD
    price_per_1m_input: float = 0.0
    price_per_1m_output: float = 0.0
    # Reasoning effort for models that support it (e.g., GPT-5.2: "low", "medium", "high", "xhigh")
    reasoning_effort: Optional[str] = None


# Token estimation constants
SYSTEM_PROMPT_TOKENS = 180  # Approximate tokens in system prompt
AVG_QUESTION_TOKENS = 350  # Average question + description tokens
AVG_OUTPUT_TOKENS = 400    # Average response tokens (reasoning + answer)


def estimate_tokens_and_cost(model_keys: list[str], num_samples: int) -> dict:
    """
    Estimate total tokens and cost for running the benchmark.
    
    Returns dict with per-model and total estimates.
    """
    estimates = {}
    total_cost = 0.0
    
    for key in model_keys:
        if key not in MODELS:
            continue
        
        config = MODELS[key]
        
        # Input tokens per sample: system prompt + question
        input_tokens_per_sample = SYSTEM_PROMPT_TOKENS + AVG_QUESTION_TOKENS
        total_input_tokens = input_tokens_per_sample * num_samples
        
        # Output tokens per sample
        total_output_tokens = AVG_OUTPUT_TOKENS * num_samples
        
        # Cost calculation
        input_cost = (total_input_tokens / 1_000_000) * config.price_per_1m_input
        output_cost = (total_output_tokens / 1_000_000) * config.price_per_1m_output
        model_cost = input_cost + output_cost
        total_cost += model_cost
        
        estimates[config.name] = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(model_cost, 4),
            "price_per_1m_input": config.price_per_1m_input,
            "price_per_1m_output": config.price_per_1m_output,
        }
    
    estimates["_total"] = {
        "total_cost_usd": round(total_cost, 2),
        "num_samples": num_samples,
        "num_models": len([k for k in model_keys if k in MODELS]),
    }
    
    return estimates


# Pre-configured models with knowledge cutoffs and pricing (USD per 1M tokens)
MODELS = {
    # ==========================================================================
    # OpenAI - GPT-5 series (latest)
    # ==========================================================================
    "gpt-5.2": ModelConfig(
        name="GPT-5.2",
        litellm_model="openai/gpt-5.2",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-10-01",
        description="OpenAI GPT-5.2 (Dec 2025)",
        temperature=1.0,  # GPT-5 only supports temperature=1
        price_per_1m_input=5.00,
        price_per_1m_output=15.00,
        reasoning_effort="medium",
    ),
    "gpt-5.2-low": ModelConfig(
        name="GPT-5.2-Low",
        litellm_model="openai/gpt-5.2",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-10-01",
        description="OpenAI GPT-5.2 with low reasoning effort",
        temperature=1.0,  # GPT-5 only supports temperature=1
        price_per_1m_input=5.00,
        price_per_1m_output=15.00,
        reasoning_effort="low",
    ),
    "gpt-5.2-high": ModelConfig(
        name="GPT-5.2-High",
        litellm_model="openai/gpt-5.2",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-10-01",
        description="OpenAI GPT-5.2 with high reasoning effort",
        temperature=1.0,  # GPT-5 only supports temperature=1
        price_per_1m_input=5.00,
        price_per_1m_output=15.00,
        reasoning_effort="high",
    ),
    "gpt-5.2-xhigh": ModelConfig(
        name="GPT-5.2-XHigh",
        litellm_model="openai/gpt-5.2",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-10-01",
        description="OpenAI GPT-5.2 with extra-high reasoning effort (exclusive to 5.2)",
        temperature=1.0,  # GPT-5 only supports temperature=1
        price_per_1m_input=5.00,
        price_per_1m_output=15.00,
        reasoning_effort="xhigh",
    ),
    "gpt-5.1": ModelConfig(
        name="GPT-5.1",
        litellm_model="openai/gpt-5.1",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-08-01",
        description="OpenAI GPT-5.1 (Nov 2025, default reasoning=none)",
        price_per_1m_input=4.00,
        price_per_1m_output=12.00,
        # GPT-5.1 with reasoning_effort=none supports temperature
    ),
    "gpt-5.1-medium": ModelConfig(
        name="GPT-5.1-Medium",
        litellm_model="openai/gpt-5.1",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-08-01",
        description="OpenAI GPT-5.1 with medium reasoning effort",
        temperature=1.0,  # GPT-5.1 with reasoning needs temperature=1
        price_per_1m_input=4.00,
        price_per_1m_output=12.00,
        reasoning_effort="medium",
    ),
    "gpt-5.1-high": ModelConfig(
        name="GPT-5.1-High",
        litellm_model="openai/gpt-5.1",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-08-01",
        description="OpenAI GPT-5.1 with high reasoning effort",
        temperature=1.0,  # GPT-5.1 with reasoning needs temperature=1
        price_per_1m_input=4.00,
        price_per_1m_output=12.00,
        reasoning_effort="high",
    ),
    # OpenAI - GPT-4o series
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        litellm_model="openai/gpt-4o",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI GPT-4o flagship multimodal",
        price_per_1m_input=2.50,
        price_per_1m_output=10.00,
        max_tokens=16384,
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        litellm_model="openai/gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI GPT-4o-mini (fast, affordable)",
        price_per_1m_input=0.15,
        price_per_1m_output=0.60,
    ),
    "gpt-5-mini": ModelConfig(
        name="GPT-5-Mini",
        litellm_model="openai/gpt-5-mini",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-08-01",
        description="OpenAI GPT-5-Mini (small, efficient)",
        price_per_1m_input=0.30,
        price_per_1m_output=1.20,
        temperature=1.0,
    ),
    "gpt-5-nano": ModelConfig(
        name="GPT-5-Nano",
        litellm_model="openai/gpt-5-nano",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-08-01",
        description="OpenAI GPT-5-Nano (smallest, most affordable)",
        price_per_1m_input=0.10,
        price_per_1m_output=0.40,
        temperature=1.0,
    ),
    # OpenAI - o1/o3 reasoning series
    "o1": ModelConfig(
        name="o1",
        litellm_model="openai/o1",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI o1 reasoning model",
        temperature=1.0,
        price_per_1m_input=15.00,
        price_per_1m_output=60.00,
    ),
    "o1-mini": ModelConfig(
        name="o1-mini",
        litellm_model="openai/o1-mini",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI o1-mini (faster reasoning)",
        temperature=1.0,
        price_per_1m_input=3.00,
        price_per_1m_output=12.00,
    ),
    "o3-mini": ModelConfig(
        name="o3-mini",
        litellm_model="openai/o3-mini",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-01-01",
        description="OpenAI o3-mini reasoning model",
        temperature=1.0,
        price_per_1m_input=1.10,
        price_per_1m_output=4.40,
    ),
    # ==========================================================================
    # Anthropic - Claude 4.5 series (latest)
    # ==========================================================================
    "claude-opus-4.5": ModelConfig(
        name="Claude-Opus-4.5",
        litellm_model="anthropic/claude-opus-4-5-20251101",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2025-04-01",
        description="Anthropic Claude Opus 4.5 (Nov 2025)",
        price_per_1m_input=15.00,
        price_per_1m_output=75.00,
    ),
    "claude-sonnet-4.5": ModelConfig(
        name="Claude-Sonnet-4.5",
        litellm_model="anthropic/claude-sonnet-4-5-20250929",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2025-04-01",
        description="Anthropic Claude Sonnet 4.5 (Sep 2025)",
        price_per_1m_input=3.00,
        price_per_1m_output=15.00,
    ),
    # Anthropic - Claude 3.5 series
    "claude-3-5-sonnet": ModelConfig(
        name="Claude-3.5-Sonnet",
        litellm_model="claude-3-5-sonnet-latest",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2024-04-01",
        description="Anthropic Claude 3.5 Sonnet",
        price_per_1m_input=3.00,
        price_per_1m_output=15.00,
    ),
     # Anthropic - Claude 3.5 series
    "claude-4-sonnet": ModelConfig(
        name="Claude-4-Sonnet",
        litellm_model="anthropic/claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2025-09-01",
        description="Anthropic Claude 4.5 Sonnet (Sep 2025)",
        price_per_1m_input=3.00,
        price_per_1m_output=15.00,
    ),
    "claude-3-5-haiku": ModelConfig(
        name="Claude-3.5-Haiku",
        litellm_model="anthropic/claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2024-04-01",
        description="Anthropic Claude 3.5 Haiku (fast)",
        price_per_1m_input=0.80,
        price_per_1m_output=4.00,
    ),
     "claude-4-5-haiku": ModelConfig(
        name="Claude-4-5-Haiku",
        litellm_model="anthropic/claude-haiku-4-5-20251001",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2025-10-01",
        description="Anthropic Claude 4.5 Haiku (Oct 2025)",
        price_per_1m_input=15.00,
        price_per_1m_output=75.00,
    ),
    # ==========================================================================
    # Together AI - Qwen
    # ==========================================================================
    "qwen-2.5-72b": ModelConfig(
        name="Qwen-2.5-72B",
        litellm_model="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-09-01",
        description="Qwen 2.5 72B via Together AI",
        price_per_1m_input=0.90,
        price_per_1m_output=0.90,
    ),
    "qwen-2.5-7b": ModelConfig(
        name="Qwen-2.5-7B",
        litellm_model="together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-09-01",
        description="Qwen 2.5 7B via Together AI",
        price_per_1m_input=0.20,
        price_per_1m_output=0.20,
    ),
    "qwen-qwq-32b": ModelConfig(
        name="Qwen-QwQ-32B",
        litellm_model="together_ai/Qwen/QwQ-32B-Preview",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-09-01",
        description="Qwen QwQ 32B reasoning model via Together AI",
        price_per_1m_input=1.20,
        price_per_1m_output=1.20,
    ),
    "qwen3-235b": ModelConfig(
        name="Qwen3-235B-Instruct",
        litellm_model="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2025-06-01",
        description="Qwen 3 235B Instruct (22B active, direct responses)",
        price_per_1m_input=2.00,
        price_per_1m_output=2.00,
    ),
    "qwen3-235b-thinking": ModelConfig(
        name="Qwen3-235B-Thinking",
        litellm_model="together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2025-06-01",
        description="Qwen 3 235B Thinking (22B active, reasoning mode)",
        price_per_1m_input=2.00,
        price_per_1m_output=2.00,
    ),
    "qwen3-32b": ModelConfig(
        name="Qwen3-32B",
        litellm_model="together_ai/Qwen/Qwen3-32B-Instruct",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2025-03-01",
        description="Qwen 3 32B via Together AI",
        price_per_1m_input=0.80,
        price_per_1m_output=0.80,
    ),
    # ==========================================================================
    # Together AI - Llama
    # ==========================================================================
    "llama-3.3-70b": ModelConfig(
        name="Llama-3.3-70B",
        litellm_model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-12-01",
        description="Meta Llama 3.3 70B via Together AI",
        price_per_1m_input=0.88,
        price_per_1m_output=0.88,
    ),
    "llama-3.1-405b": ModelConfig(
        name="Llama-3.1-405B",
        litellm_model="together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-03-01",
        description="Meta Llama 3.1 405B via Together AI",
        price_per_1m_input=3.50,
        price_per_1m_output=3.50,
    ),
    "llama-3.1-8b": ModelConfig(
        name="Llama-3.1-8B",
        litellm_model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-03-01",
        description="Meta Llama 3.1 8B via Together AI",
        price_per_1m_input=0.18,
        price_per_1m_output=0.18,
    ),
    # ==========================================================================
    # Fireworks AI - DeepSeek
    # ==========================================================================
    "deepseek-v3": ModelConfig(
        name="DeepSeek-V3",
        litellm_model="fireworks_ai/accounts/fireworks/models/deepseek-v3",
        provider=ModelProvider.FIREWORKS,
        knowledge_cutoff="2024-11-01",
        description="DeepSeek V3 via Fireworks AI",
        price_per_1m_input=0.90,
        price_per_1m_output=0.90,
        max_tokens=4096,
    ),
    "deepseek-v3.1": ModelConfig(
        name="DeepSeek-V3.1",
        litellm_model="fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus",
        provider=ModelProvider.FIREWORKS,
        knowledge_cutoff="2025-06-01",
        description="DeepSeek V3.1 via Fireworks AI",
        price_per_1m_input=1.00,
        price_per_1m_output=1.00,
        max_tokens=4096,
    ),
    "deepseek-v3.2": ModelConfig(
        name="DeepSeek-V3.2",
        litellm_model="fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        provider=ModelProvider.FIREWORKS,
        knowledge_cutoff="2025-10-01",
        description="DeepSeek V3.2 (GPT-5 level, 685B MoE) via Fireworks AI",
        price_per_1m_input=1.20,
        price_per_1m_output=1.20,
        max_tokens=4096,
    ),
    "deepseek-r1": ModelConfig(
        name="DeepSeek-R1",
        litellm_model="fireworks_ai/accounts/fireworks/models/deepseek-r1",
        provider=ModelProvider.FIREWORKS,
        knowledge_cutoff="2024-11-01",
        description="DeepSeek R1 reasoning model via Fireworks AI",
        price_per_1m_input=3.00,
        price_per_1m_output=7.00,
        max_tokens=4096,
    ),
    # ==========================================================================
    # Fireworks AI - Kimi (Moonshot AI)
    # ==========================================================================
    "kimi-k2": ModelConfig(
        name="Kimi-K2",
        litellm_model="fireworks_ai/accounts/fireworks/models/kimi-k2-thinking",
        provider=ModelProvider.FIREWORKS,
        knowledge_cutoff="2025-06-01",
        description="Kimi K2 (1T params, 32B active) via Fireworks AI",
        price_per_1m_input=1.50,
        price_per_1m_output=1.50,
        max_tokens=4096,
    ),
    # ==========================================================================
    # Together AI - Mistral
    # ==========================================================================
    "mistral-large": ModelConfig(
        name="Mistral-Large",
        litellm_model="together_ai/mistralai/Mistral-Large-Instruct-2407",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-07-01",
        description="Mistral Large via Together AI",
        price_per_1m_input=2.00,
        price_per_1m_output=6.00,
    ),
    # ==========================================================================
    # Google Gemini
    # ==========================================================================
    "gemini-3-pro": ModelConfig(
        name="Gemini-3-Pro",
        litellm_model="gemini/gemini-3.0-pro",
        provider=ModelProvider.GOOGLE,
        knowledge_cutoff="2025-01-01",
        description="Google Gemini 3 Pro (Nov 2025, latest flagship)",
        price_per_1m_input=2.50,
        price_per_1m_output=10.00,
    ),
    "gemini-3-flash": ModelConfig(
        name="Gemini-3-Flash",
        litellm_model="gemini/gemini-3.0-flash",
        provider=ModelProvider.GOOGLE,
        knowledge_cutoff="2025-01-01",
        description="Google Gemini 3 Flash (Nov 2025, fast)",
        price_per_1m_input=0.15,
        price_per_1m_output=0.60,
    ),
    "gemini-2.5-pro": ModelConfig(
        name="Gemini-2.5-Pro",
        litellm_model="gemini/gemini-2.5-pro-preview-06-05",
        provider=ModelProvider.GOOGLE,
        knowledge_cutoff="2025-01-01",
        description="Google Gemini 2.5 Pro",
        price_per_1m_input=1.25,
        price_per_1m_output=10.00,
    ),
    "gemini-2.0-flash": ModelConfig(
        name="Gemini-2.0-Flash",
        litellm_model="gemini/gemini-2.0-flash-exp",
        provider=ModelProvider.GOOGLE,
        knowledge_cutoff="2024-08-01",
        description="Google Gemini 2.0 Flash",
        price_per_1m_input=0.10,
        price_per_1m_output=0.40,
    ),
}


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    # Use pre-cleaned KalshiBench dataset (already deduplicated)
    dataset_name: str = "2084Collective/kalshibench-v2"
    knowledge_cutoff: str | None = None  # Auto-computed from models if None
    num_samples: int = 200
    output_dir: str = "kalshibench_results"
    seed: int = 42
    max_concurrent: int = 10  # Max concurrent API calls
    confidence_bins: int = 10  # For calibration analysis
    # Legacy: set to True to use raw dataset with deduplication
    use_raw_dataset: bool = False
    raw_dataset_name: str = "2084Collective/prediction-markets-historical-v5-cleaned"


def get_max_knowledge_cutoff(model_keys: list[str]) -> str:
    """
    Get the latest knowledge cutoff date from a list of models.
    This ensures we only test on questions that resolve AFTER all models' training data.
    """
    cutoffs = []
    for key in model_keys:
        if key in MODELS:
            cutoffs.append(MODELS[key].knowledge_cutoff)
    
    if not cutoffs:
        return "2024-01-01"  # Fallback default
    
    # Return the maximum (latest) date
    return max(cutoffs)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class MarketQuestion:
    """A single prediction market question."""
    id: str
    question: str
    description: str
    category: str
    close_time: str
    ground_truth: Literal["yes", "no"]
    market_probability: Optional[float] = None  # Kalshi market price if available


@dataclass
class ModelPrediction:
    """A single model prediction."""
    question_id: str
    predicted_answer: Optional[Literal["yes", "no"]]
    predicted_probability: float  # P(yes)
    reasoning: str
    raw_output: str
    latency_ms: float
    error: Optional[str] = None
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass 
class EvaluationResult:
    """Complete evaluation results for a single model."""
    model_name: str
    model_config: dict
    timestamp: str
    num_samples: int
    
    # Classification metrics
    accuracy: float
    precision_yes: float
    recall_yes: float
    f1_yes: float
    precision_no: float
    recall_no: float
    f1_no: float
    macro_f1: float
    
    # Calibration metrics
    brier_score: float
    brier_skill_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Adaptive Calibration Error
    log_loss: float
    
    # Confidence analysis
    avg_confidence: float
    avg_confidence_when_correct: Optional[float]
    avg_confidence_when_wrong: Optional[float]
    overconfidence_rate_70: Optional[float]
    overconfidence_rate_80: Optional[float]
    overconfidence_rate_90: Optional[float]
    
    # Reliability diagram data
    reliability_diagram: list
    
    # Per-category breakdown
    per_category: dict
    
    # Confusion matrix
    confusion_matrix: dict
    
    # Meta
    parse_rate: float
    avg_latency_ms: float
    
    # Token usage and cost
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    
    # Raw predictions for further analysis
    predictions: list = field(default_factory=list)


# ==============================================================================
# Abstract Base Classes
# ==============================================================================

class BaseModel(ABC):
    """Abstract base class for model inference."""
    
    @abstractmethod
    async def predict(self, question: MarketQuestion) -> ModelPrediction:
        """Generate a prediction for a market question."""
        pass
    
    @abstractmethod
    def get_config(self) -> dict:
        """Return model configuration."""
        pass


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass
    
    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Whether higher values indicate better performance."""
        pass
    
    @abstractmethod
    def compute(
        self, 
        predictions: list[ModelPrediction], 
        questions: list[MarketQuestion]
    ) -> float:
        """Compute the metric value."""
        pass


class BaseDataLoader(ABC):
    """Abstract base class for data loading."""
    
    @abstractmethod
    def load(self) -> list[MarketQuestion]:
        """Load and return market questions."""
        pass


# ==============================================================================
# Implementations
# ==============================================================================

class LiteLLMModel(BaseModel):
    """Model implementation using LiteLLM for unified API access."""
    
    SYSTEM_PROMPT = """You are an expert forecaster evaluating prediction market questions. 
Given a question and its description, predict whether the outcome will be "yes" or "no".

You must respond in this exact format:
<think>
[Your reasoning about the prediction, considering base rates, relevant factors, and uncertainty]
</think>
<answer>[yes or no]</answer>
<confidence>[a number from 0 to 100 representing your confidence that the answer is "yes"]</confidence>

Be calibrated: if you're 70% confident, you should be correct about 70% of the time on similar questions."""

    def __init__(self, config: ModelConfig):
        self.config = config
    
    def get_config(self) -> dict:
        return self.config.model_dump(mode="json")
    
    def _format_prompt(self, question: MarketQuestion) -> str:
        return f"""Question: {question.question}

Description: {question.description}

Based on the information provided, predict whether this will resolve to "yes" or "no"."""

    def _parse_response(self, text: str) -> tuple[Optional[str], float, str]:
        """Parse model response to extract answer, confidence, and reasoning."""
        # Extract answer
        answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE)
        answer = answer_match.group(1).lower() if answer_match else None
        
        # Extract confidence
        conf_match = re.search(r'<confidence>\s*(\d+(?:\.\d+)?)\s*</confidence>', text, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1)) / 100.0  # Convert to 0-1
            confidence = max(0.0, min(1.0, confidence))  # Clamp
        else:
            # Fallback: use 0.5 if no confidence provided
            confidence = 0.5
        
        # If answer is "no", convert confidence to P(yes)
        if answer == "no":
            confidence = 1.0 - confidence
        
        # Extract reasoning
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        # Fallback answer extraction
        if answer is None:
            text_lower = text.lower()
            if 'the answer is yes' in text_lower:
                answer = 'yes'
            elif 'the answer is no' in text_lower:
                answer = 'no'
            else:
                # Count occurrences
                yes_count = text_lower.count(' yes')
                no_count = text_lower.count(' no')
                if yes_count > no_count:
                    answer = 'yes'
                elif no_count > yes_count:
                    answer = 'no'
        
        return answer, confidence, reasoning

    async def predict(self, question: MarketQuestion) -> ModelPrediction:
        """Generate prediction using LiteLLM."""
        start_time = datetime.now()
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._format_prompt(question)},
        ]
        
        # Call acompletion with optional reasoning_effort
        response = await acompletion(
            model=self.config.litellm_model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            reasoning_effort=self.config.reasoning_effort,  # type: ignore[arg-type]
        )
        
        raw_output = response.choices[0].message.content
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
        
        answer, confidence, reasoning = self._parse_response(raw_output)
        
        # Cast answer to the expected type
        valid_answer: Optional[Literal["yes", "no"]] = None
        if answer in ("yes", "no"):
            valid_answer = answer  # type: ignore
        
        return ModelPrediction(
            question_id=question.id,
            predicted_answer=valid_answer,
            predicted_probability=confidence,
            reasoning=reasoning,
            raw_output=raw_output,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
       

class KalshiDataLoader(BaseDataLoader):
    """Load prediction market data from HuggingFace dataset.
    
    By default, loads from the pre-cleaned KalshiBench dataset which is already
    deduplicated. Set config.use_raw_dataset=True to use the raw dataset with
    on-the-fly deduplication.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def load(self) -> list[MarketQuestion]:
        """Load and filter market questions."""
        if self.config.use_raw_dataset:
            return self._load_raw_with_dedup()
        else:
            return self._load_cleaned()
    
    def _load_cleaned(self) -> list[MarketQuestion]:
        """Load from pre-cleaned KalshiBench dataset (recommended)."""
        logger.info(f"📥 Loading KalshiBench: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name, split="train")
        logger.info(f"   Dataset size: {len(dataset):,} (pre-cleaned, deduplicated)")
        
        # Filter by date (post-knowledge-cutoff) - this is the only runtime filter needed
        if self.config.knowledge_cutoff:
            logger.info(f"🔍 Filtering by date >= {self.config.knowledge_cutoff}")
            def filter_by_date(example):
                close_time = example.get("close_time", "")
                return close_time and close_time >= self.config.knowledge_cutoff
            
            dataset = dataset.filter(filter_by_date)
            logger.info(f"   After date filter: {len(dataset):,}")
        
        # Convert to MarketQuestion objects
        logger.info("📝 Converting to MarketQuestion objects")
        questions = []
        for item in tqdm(dataset, desc="Loading questions", unit="q"):
            # Pre-cleaned dataset uses 'ground_truth' field
            ground_truth = item.get("ground_truth", "")
            if ground_truth not in ["yes", "no"]:
                # Fallback for raw dataset format
                ground_truth = (item.get("winning_outcome") or "").lower()
                if ground_truth not in ["yes", "no"]:
                    continue
            
            questions.append(MarketQuestion(
                id=item.get("id") or f"kalshi_{len(questions)}",
                question=item.get("question", ""),
                description=(item.get("description") or "")[:2000],
                category=item.get("category", "unknown"),
                close_time=item.get("close_time", ""),
                ground_truth=ground_truth,
                market_probability=item.get("market_probability") or item.get("last_price"),
            ))
        
        logger.info(f"   Valid questions: {len(questions):,}")
        
        # Sample if needed
        questions = self._sample_questions(questions)
        
        # Show summary
        self._log_summary(questions)
        return questions
    
    def _load_raw_with_dedup(self) -> list[MarketQuestion]:
        """Load from raw dataset with on-the-fly deduplication (legacy)."""
        logger.info(f"📥 Loading raw dataset: {self.config.raw_dataset_name}")
        logger.info("   ⚠️  Using legacy mode with on-the-fly deduplication")
        dataset = load_dataset(self.config.raw_dataset_name, split="train")
        logger.info(f"   Raw dataset size: {len(dataset):,}")
        
        # Filter by date (post-knowledge-cutoff)
        if self.config.knowledge_cutoff:
            logger.info(f"🔍 Filtering by date >= {self.config.knowledge_cutoff}")
            def filter_by_date(example):
                close_time = example.get("close_time", "")
                return close_time and close_time >= self.config.knowledge_cutoff
            
            dataset = dataset.filter(filter_by_date)
            logger.info(f"   After date filter: {len(dataset):,}")
        
        # Deduplicate by series_ticker
        logger.info("🔄 Deduplicating by series_ticker")
        seen_tickers: set[str] = set()
        def deduplicate(example):
            ticker = example.get("series_ticker", "")
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                return True
            return False
        
        dataset = dataset.filter(deduplicate)
        logger.info(f"   After deduplication: {len(dataset):,}")
        
        # Convert to MarketQuestion objects
        logger.info("📝 Converting to MarketQuestion objects")
        questions = []
        for item in tqdm(dataset, desc="Processing questions", unit="q"):
            winning_outcome = (item.get("winning_outcome") or "").lower()
            if winning_outcome not in ["yes", "no"]:
                continue
            
            questions.append(MarketQuestion(
                id=f"kalshi_{len(questions)}_{item.get('series_ticker', 'unknown')}",
                question=item.get("question", ""),
                description=(item.get("description") or "")[:2000],
                category=item.get("category", "unknown"),
                close_time=item.get("close_time", ""),
                ground_truth=winning_outcome,
                market_probability=item.get("last_price"),
            ))
        
        logger.info(f"   Valid questions: {len(questions):,}")
        
        # Sample if needed
        questions = self._sample_questions(questions)
        
        # Show summary
        self._log_summary(questions)
        return questions
    
    def _sample_questions(self, questions: list[MarketQuestion]) -> list[MarketQuestion]:
        """Sample questions if we have more than num_samples."""
        if len(questions) > self.config.num_samples:
            np.random.seed(self.config.seed)
            logger.info(f"🎲 Sampling {self.config.num_samples:,} questions from {len(questions):,} (seed={self.config.seed})")
            indices = np.random.choice(len(questions), self.config.num_samples, replace=False)
            questions = [questions[i] for i in indices]
        return questions
    
    def _log_summary(self, questions: list[MarketQuestion]) -> None:
        """Log category distribution summary."""
        categories: dict[str, int] = defaultdict(int)
        for q in questions:
            categories[q.category] += 1
        
        logger.info(f"   Category distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"      {cat}: {count}")
        if len(categories) > 5:
            logger.info(f"      ... and {len(categories) - 5} more categories")
        
        logger.info(f"✅ Loaded {len(questions):,} questions")


# ==============================================================================
# Dataset Analysis Models (Pydantic)
# ==============================================================================

class DateRangeStats(PydanticBaseModel):
    """Temporal range of the dataset."""
    earliest: str | None = None
    latest: str | None = None
    span_days: int = 0


class TimeHorizonStats(PydanticBaseModel):
    """Statistics about prediction time horizons."""
    mean_days_ahead: float = 0.0
    median_days_ahead: float = 0.0
    std_days_ahead: float = 0.0
    min_days_ahead: int = 0
    max_days_ahead: int = 0


class DistributionStats(PydanticBaseModel):
    """Basic statistical distribution metrics."""
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: int = 0
    max: int = 0


class GroundTruthDistribution(PydanticBaseModel):
    """Overall ground truth label distribution."""
    yes: int = 0
    no: int = 0
    yes_rate: float = 0.0


class CategoryGroundTruth(PydanticBaseModel):
    """Ground truth distribution for a single category."""
    yes: int = 0
    no: int = 0
    total: int = 0
    yes_rate: float = 0.0


class MonthGroundTruth(PydanticBaseModel):
    """Ground truth distribution for a single month."""
    yes_rate: float = 0.0
    total: int = 0


class ReliabilityBin(PydanticBaseModel):
    """Single bin in a reliability diagram."""
    bin: str
    avg_confidence: float
    avg_accuracy: float
    count: int


class MarketCalibration(PydanticBaseModel):
    """Calibration metrics for market prices."""
    brier_score: float
    brier_skill_score: float
    ece: float
    log_loss: float
    accuracy: float
    base_rate: float


class UncertaintyDistribution(PydanticBaseModel):
    """Distribution of market uncertainty levels."""
    high_confidence: int = Field(0, description="Count of predictions with prob <0.2 or >0.8")
    medium_confidence: int = Field(0, description="Count of predictions with prob 0.2-0.4 or 0.6-0.8")
    low_confidence: int = Field(0, description="Count of predictions with prob 0.4-0.6")


class TemporalAnalysis(PydanticBaseModel):
    """Complete temporal analysis results."""
    date_range: DateRangeStats
    by_month: dict[str, int] = Field(default_factory=dict)
    time_horizon: TimeHorizonStats | None = None


class GroundTruthAnalysis(PydanticBaseModel):
    """Complete ground truth analysis results."""
    distribution: GroundTruthDistribution
    by_category: dict[str, CategoryGroundTruth] = Field(default_factory=dict)
    by_month: dict[str, MonthGroundTruth] = Field(default_factory=dict)


class CategoryAnalysis(PydanticBaseModel):
    """Complete category analysis results."""
    distribution: dict[str, int] = Field(default_factory=dict)
    entropy: float = 0.0
    normalized_entropy: float = 0.0
    num_categories: int = 0
    largest_category: str | None = None
    smallest_category: str | None = None


class ComplexityAnalysis(PydanticBaseModel):
    """Question complexity analysis results."""
    question_length: DistributionStats
    description_length: DistributionStats
    vocabulary_size: int = 0
    avg_words: float = 0.0


class MarketBaselineAnalysis(PydanticBaseModel):
    """Complete market baseline analysis."""
    coverage: float = 0.0
    calibration: MarketCalibration | None = None
    reliability_diagram: list[ReliabilityBin] = Field(default_factory=list)
    uncertainty_distribution: UncertaintyDistribution | None = None
    high_confidence_accuracy: float | None = None
    uncertain_accuracy: float | None = None


class DatasetAnalysis(PydanticBaseModel):
    """Comprehensive dataset analysis for paper generation."""
    # Basic stats
    total_questions: int
    knowledge_cutoff: str
    
    # Temporal distribution
    date_range: DateRangeStats
    temporal_distribution: dict[str, int] = Field(default_factory=dict)
    time_horizon_stats: TimeHorizonStats | None = None
    
    # Ground truth analysis
    ground_truth_distribution: GroundTruthDistribution
    ground_truth_by_category: dict[str, CategoryGroundTruth] = Field(default_factory=dict)
    ground_truth_by_month: dict[str, MonthGroundTruth] = Field(default_factory=dict)
    
    # Category analysis
    category_distribution: dict[str, int] = Field(default_factory=dict)
    category_entropy: float = 0.0
    num_categories: int = 0
    largest_category: str | None = None
    smallest_category: str | None = None
    
    # Question complexity
    question_length_stats: DistributionStats
    description_length_stats: DistributionStats
    total_vocabulary_size: int = 0
    avg_words_per_question: float = 0.0
    
    # Market baseline
    market_coverage: float = 0.0
    market_calibration: MarketCalibration | None = None
    market_reliability_diagram: list[ReliabilityBin] = Field(default_factory=list)
    market_uncertainty_distribution: UncertaintyDistribution | None = None
    high_confidence_market_accuracy: float | None = None
    uncertain_market_accuracy: float | None = None


class DatasetAnalyzer:
    """Analyze dataset characteristics for paper generation."""
    
    def __init__(self, questions: list[MarketQuestion], config: BenchmarkConfig):
        self.questions = questions
        self.config = config
    
    def analyze(self) -> DatasetAnalysis:
        """Run full dataset analysis."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 DATASET ANALYSIS")
        logger.info("=" * 60)
        
        # Basic stats
        total = len(self.questions)
        
        # Temporal analysis
        temporal = self._analyze_temporal()
        logger.info(f"   Date range: {temporal.date_range.earliest} to {temporal.date_range.latest}")
        logger.info(f"   Span: {temporal.date_range.span_days} days")
        
        # Ground truth analysis
        gt_analysis = self._analyze_ground_truth()
        logger.info(f"   Ground truth: {gt_analysis.distribution.yes_rate:.1%} Yes, {1-gt_analysis.distribution.yes_rate:.1%} No")
        
        # Category analysis
        cat_analysis = self._analyze_categories()
        logger.info(f"   Categories: {cat_analysis.num_categories} unique")
        if cat_analysis.largest_category:
            logger.info(f"   Largest: {cat_analysis.largest_category} ({cat_analysis.distribution.get(cat_analysis.largest_category, 0)})")
        
        # Complexity analysis
        complexity = self._analyze_complexity()
        logger.info(f"   Avg question length: {complexity.question_length.mean:.0f} chars")
        logger.info(f"   Vocabulary size: {complexity.vocabulary_size:,} unique words")
        
        # Market baseline
        market = self._analyze_market_baseline()
        if market.coverage > 0:
            logger.info(f"   Market coverage: {market.coverage:.1%}")
            if market.calibration:
                logger.info(f"   Market Brier: {market.calibration.brier_score:.4f}")
                logger.info(f"   Market accuracy: {market.calibration.accuracy:.1%}")
        
        return DatasetAnalysis(
            total_questions=total,
            knowledge_cutoff=self.config.knowledge_cutoff or "unknown",
            date_range=temporal.date_range,
            temporal_distribution=temporal.by_month,
            time_horizon_stats=temporal.time_horizon,
            ground_truth_distribution=gt_analysis.distribution,
            ground_truth_by_category=gt_analysis.by_category,
            ground_truth_by_month=gt_analysis.by_month,
            category_distribution=cat_analysis.distribution,
            category_entropy=cat_analysis.entropy,
            num_categories=cat_analysis.num_categories,
            largest_category=cat_analysis.largest_category,
            smallest_category=cat_analysis.smallest_category,
            question_length_stats=complexity.question_length,
            description_length_stats=complexity.description_length,
            total_vocabulary_size=complexity.vocabulary_size,
            avg_words_per_question=complexity.avg_words,
            market_coverage=market.coverage,
            market_calibration=market.calibration,
            market_reliability_diagram=market.reliability_diagram,
            market_uncertainty_distribution=market.uncertainty_distribution,
            high_confidence_market_accuracy=market.high_confidence_accuracy,
            uncertain_market_accuracy=market.uncertain_accuracy,
        )
    
    def _analyze_temporal(self) -> TemporalAnalysis:
        """Analyze temporal distribution of questions."""
        close_times = [q.close_time for q in self.questions if q.close_time]
        
        if not close_times:
            return TemporalAnalysis(
                date_range=DateRangeStats(),
                by_month={},
                time_horizon=None,
            )
        
        earliest = min(close_times)
        latest = max(close_times)
        
        # Calculate span
        earliest_dt = datetime.fromisoformat(earliest.replace('Z', '+00:00').split('T')[0])
        latest_dt = datetime.fromisoformat(latest.replace('Z', '+00:00').split('T')[0])
        span_days = (latest_dt - earliest_dt).days
        
        # Distribution by month
        by_month: dict[str, int] = defaultdict(int)
        for ct in close_times:
            month = ct[:7]  # YYYY-MM
            by_month[month] += 1
        
        # Time horizon analysis (days from knowledge cutoff to close)
        horizons: list[int] = []
        if self.config.knowledge_cutoff:
            cutoff_dt = datetime.fromisoformat(self.config.knowledge_cutoff)
            for ct in close_times:
                close_dt = datetime.fromisoformat(ct.replace('Z', '+00:00').split('T')[0])
                days_ahead = (close_dt - cutoff_dt).days
                if days_ahead > 0:
                    horizons.append(days_ahead)
        
        horizon_stats = None
        if horizons:
            horizon_stats = TimeHorizonStats(
                mean_days_ahead=float(np.mean(horizons)),
                median_days_ahead=float(np.median(horizons)),
                std_days_ahead=float(np.std(horizons)),
                min_days_ahead=int(min(horizons)),
                max_days_ahead=int(max(horizons)),
            )
        
        return TemporalAnalysis(
            date_range=DateRangeStats(earliest=earliest, latest=latest, span_days=span_days),
            by_month=dict(sorted(by_month.items())),
            time_horizon=horizon_stats,
        )
    
    def _analyze_ground_truth(self) -> GroundTruthAnalysis:
        """Analyze ground truth distribution and biases."""
        yes_count = sum(1 for q in self.questions if q.ground_truth == "yes")
        no_count = len(self.questions) - yes_count
        yes_rate = yes_count / len(self.questions) if self.questions else 0.0
        
        # By category
        by_category_raw: dict[str, dict[str, int]] = defaultdict(lambda: {'yes': 0, 'no': 0})
        for q in self.questions:
            if q.ground_truth == "yes":
                by_category_raw[q.category]['yes'] += 1
            else:
                by_category_raw[q.category]['no'] += 1
        
        by_category: dict[str, CategoryGroundTruth] = {}
        for cat, counts in by_category_raw.items():
            total = counts['yes'] + counts['no']
            by_category[cat] = CategoryGroundTruth(
                yes=counts['yes'],
                no=counts['no'],
                total=total,
                yes_rate=counts['yes'] / total if total > 0 else 0.0,
            )
        
        # By month
        by_month_raw: dict[str, dict[str, int]] = defaultdict(lambda: {'yes': 0, 'total': 0})
        for q in self.questions:
            month = q.close_time[:7] if q.close_time else 'unknown'
            by_month_raw[month]['total'] += 1
            if q.ground_truth == "yes":
                by_month_raw[month]['yes'] += 1
        
        by_month: dict[str, MonthGroundTruth] = {}
        for month, counts in sorted(by_month_raw.items()):
            by_month[month] = MonthGroundTruth(
                yes_rate=counts['yes'] / counts['total'] if counts['total'] > 0 else 0.0,
                total=counts['total'],
            )
        
        return GroundTruthAnalysis(
            distribution=GroundTruthDistribution(yes=yes_count, no=no_count, yes_rate=yes_rate),
            by_category=by_category,
            by_month=by_month,
        )
    
    def _analyze_categories(self) -> CategoryAnalysis:
        """Analyze category distribution and diversity."""
        category_counts: dict[str, int] = defaultdict(int)
        for q in self.questions:
            category_counts[q.category] += 1
        
        # Shannon entropy
        total = len(self.questions)
        probs = [count / total for count in category_counts.values()]
        entropy = float(-sum(p * np.log2(p) for p in probs if p > 0))
        max_entropy = float(np.log2(len(category_counts))) if category_counts else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
        
        return CategoryAnalysis(
            distribution=dict(sorted_cats),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            num_categories=len(category_counts),
            largest_category=sorted_cats[0][0] if sorted_cats else None,
            smallest_category=sorted_cats[-1][0] if sorted_cats else None,
        )
    
    def _analyze_complexity(self) -> ComplexityAnalysis:
        """Analyze question complexity metrics."""
        question_lengths = [len(q.question) for q in self.questions]
        description_lengths = [len(q.description) for q in self.questions]
        
        # Vocabulary analysis
        all_words: set[str] = set()
        word_counts: list[int] = []
        for q in self.questions:
            words = re.findall(r'\b\w+\b', (q.question + " " + q.description).lower())
            all_words.update(words)
            word_counts.append(len(words))
        
        def make_stats(arr: list[int]) -> DistributionStats:
            if not arr:
                return DistributionStats()
            return DistributionStats(
                mean=float(np.mean(arr)),
                median=float(np.median(arr)),
                std=float(np.std(arr)),
                min=int(min(arr)),
                max=int(max(arr)),
            )
        
        return ComplexityAnalysis(
            question_length=make_stats(question_lengths),
            description_length=make_stats(description_lengths),
            vocabulary_size=len(all_words),
            avg_words=float(np.mean(word_counts)) if word_counts else 0.0,
        )
    
    def _analyze_market_baseline(self) -> MarketBaselineAnalysis:
        """Analyze market prices as a baseline predictor."""
        # Get questions with market prices
        with_prices = [(q, q.market_probability) for q in self.questions 
                       if q.market_probability is not None]
        
        coverage = len(with_prices) / len(self.questions) if self.questions else 0.0
        
        if not with_prices:
            return MarketBaselineAnalysis(coverage=coverage)
        
        # Extract arrays
        probs = np.array([p for _, p in with_prices])
        actuals = np.array([1.0 if q.ground_truth == "yes" else 0.0 for q, _ in with_prices])
        predictions = (probs >= 0.5).astype(int)
        correct = predictions == actuals
        
        # Brier score
        brier = float(np.mean((probs - actuals) ** 2))
        
        # Base rate for skill score
        base_rate = float(np.mean(actuals))
        brier_clim = float(np.mean((base_rate - actuals) ** 2))
        brier_skill = 1 - (brier / brier_clim) if brier_clim > 0 else 0.0
        
        # Accuracy
        accuracy = float(np.mean(correct))
        
        # ECE
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        reliability_diagram: list[ReliabilityBin] = []
        
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if np.sum(mask) > 0:
                bin_conf = float(np.mean(probs[mask]))
                bin_acc = float(np.mean(actuals[mask]))
                bin_count = int(np.sum(mask))
                ece += bin_count * abs(bin_conf - bin_acc)
                reliability_diagram.append(ReliabilityBin(
                    bin=f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    avg_confidence=bin_conf,
                    avg_accuracy=bin_acc,
                    count=bin_count,
                ))
        ece = ece / len(probs)
        
        # Log loss
        eps = 1e-15
        probs_clip = np.clip(probs, eps, 1 - eps)
        log_loss = float(-np.mean(actuals * np.log(probs_clip) + (1 - actuals) * np.log(1 - probs_clip)))
        
        # Uncertainty distribution
        uncertainty_dist = UncertaintyDistribution(
            high_confidence=int(np.sum((probs < 0.2) | (probs > 0.8))),
            medium_confidence=int(np.sum(((probs >= 0.2) & (probs < 0.4)) | ((probs > 0.6) & (probs <= 0.8)))),
            low_confidence=int(np.sum((probs >= 0.4) & (probs <= 0.6))),
        )
        
        # High confidence accuracy
        high_conf_mask = (probs > 0.8) | (probs < 0.2)
        high_conf_acc = float(np.mean(correct[high_conf_mask])) if np.sum(high_conf_mask) > 0 else None
        
        # Uncertain accuracy  
        uncertain_mask = (probs >= 0.4) & (probs <= 0.6)
        uncertain_acc = float(np.mean(correct[uncertain_mask])) if np.sum(uncertain_mask) > 0 else None
        
        return MarketBaselineAnalysis(
            coverage=coverage,
            calibration=MarketCalibration(
                brier_score=brier,
                brier_skill_score=brier_skill,
                ece=ece,
                log_loss=log_loss,
                accuracy=accuracy,
                base_rate=base_rate,
            ),
            reliability_diagram=reliability_diagram,
            uncertainty_distribution=uncertainty_dist,
            high_confidence_accuracy=high_conf_acc,
            uncertain_accuracy=uncertain_acc,
        )


# ==============================================================================
# Metrics Implementation
# ==============================================================================

class MetricsCalculator:
    """Calculate all evaluation metrics."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def compute_all(
        self,
        predictions: list[ModelPrediction],
        questions: list[MarketQuestion],
    ) -> dict:
        """Compute all metrics."""
        # Build aligned lists
        question_map = {q.id: q for q in questions}
        
        valid_preds = []
        valid_questions = []
        all_probs = []
        all_actuals = []
        all_correct = []
        
        for pred in predictions:
            q = question_map.get(pred.question_id)
            if q and pred.predicted_answer in ["yes", "no"]:
                valid_preds.append(pred)
                valid_questions.append(q)
                all_probs.append(pred.predicted_probability)
                all_actuals.append(1.0 if q.ground_truth == "yes" else 0.0)
                all_correct.append(pred.predicted_answer == q.ground_truth)
        
        probs = np.array(all_probs)
        actuals = np.array(all_actuals)
        correct = np.array(all_correct)
        
        metrics = {}
        
        # Classification metrics
        metrics.update(self._classification_metrics(valid_preds, valid_questions))
        
        # Calibration metrics
        metrics.update(self._calibration_metrics(probs, actuals, correct))
        
        # Confidence analysis
        metrics.update(self._confidence_analysis(probs, actuals, correct))
        
        # Reliability diagram
        metrics["reliability_diagram"] = self._reliability_diagram(probs, actuals)
        
        # Per-category breakdown
        metrics["per_category"] = self._per_category(valid_preds, valid_questions)
        
        # Meta
        metrics["parse_rate"] = len(valid_preds) / len(predictions) if predictions else 0
        metrics["avg_latency_ms"] = np.mean([p.latency_ms for p in predictions])
        
        return metrics
    
    def _classification_metrics(
        self, 
        predictions: list[ModelPrediction],
        questions: list[MarketQuestion],
    ) -> dict:
        """Compute classification metrics."""
        tp = fp = fn = tn = 0
        
        for pred, q in zip(predictions, questions):
            if pred.predicted_answer == "yes" and q.ground_truth == "yes":
                tp += 1
            elif pred.predicted_answer == "yes" and q.ground_truth == "no":
                fp += 1
            elif pred.predicted_answer == "no" and q.ground_truth == "yes":
                fn += 1
            else:
                tn += 1
        
        total = tp + fp + fn + tn
        
        precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes) if (precision_yes + recall_yes) > 0 else 0
        
        precision_no = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_no = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_no = 2 * precision_no * recall_no / (precision_no + recall_no) if (precision_no + recall_no) > 0 else 0
        
        return {
            "accuracy": (tp + tn) / total if total > 0 else 0,
            "precision_yes": precision_yes,
            "recall_yes": recall_yes,
            "f1_yes": f1_yes,
            "precision_no": precision_no,
            "recall_no": recall_no,
            "f1_no": f1_no,
            "macro_f1": (f1_yes + f1_no) / 2,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    
    def _calibration_metrics(
        self,
        probs: np.ndarray,
        actuals: np.ndarray,
        correct: np.ndarray,
    ) -> dict:
        """Compute calibration metrics."""
        if len(probs) == 0:
            return {
                "brier_score": None,
                "brier_skill_score": None,
                "ece": None,
                "mce": None,
                "ace": None,
                "log_loss": None,
            }
        
        # Brier Score
        brier = np.mean((probs - actuals) ** 2)
        
        # Brier Skill Score (vs climatology)
        base_rate = np.mean(actuals)
        brier_clim = np.mean((base_rate - actuals) ** 2)
        brier_skill = 1 - (brier / brier_clim) if brier_clim > 0 else 0
        
        # Log Loss
        eps = 1e-15
        probs_clip = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean(actuals * np.log(probs_clip) + (1 - actuals) * np.log(1 - probs_clip))
        
        # ECE, MCE
        n_bins = self.config.confidence_bins
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0
        mce = 0
        
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if np.sum(mask) > 0:
                bin_conf = np.mean(probs[mask])
                bin_acc = np.mean(actuals[mask])
                bin_ce = abs(bin_conf - bin_acc)
                ece += np.sum(mask) * bin_ce
                mce = max(mce, bin_ce)
        
        ece = ece / len(probs)
        
        # ACE (Adaptive Calibration Error - equal mass bins)
        sorted_idx = np.argsort(probs)
        n_per_bin = len(probs) // n_bins
        ace = 0
        for i in range(n_bins):
            start = i * n_per_bin
            end = start + n_per_bin if i < n_bins - 1 else len(probs)
            bin_idx = sorted_idx[start:end]
            if len(bin_idx) > 0:
                ace += len(bin_idx) * abs(np.mean(probs[bin_idx]) - np.mean(actuals[bin_idx]))
        ace = ace / len(probs)
        
        return {
            "brier_score": float(brier),
            "brier_skill_score": float(brier_skill),
            "ece": float(ece),
            "mce": float(mce),
            "ace": float(ace),
            "log_loss": float(log_loss),
        }
    
    def _confidence_analysis(
        self,
        probs: np.ndarray,
        actuals: np.ndarray,
        correct: np.ndarray,
    ) -> dict:
        """Analyze confidence patterns."""
        if len(probs) == 0:
            return {
                "avg_confidence": None,
                "avg_confidence_when_correct": None,
                "avg_confidence_when_wrong": None,
                "overconfidence_rate_70": None,
                "overconfidence_rate_80": None,
                "overconfidence_rate_90": None,
            }
        
        # Average confidence (distance from 0.5)
        conf = np.abs(probs - 0.5) + 0.5
        avg_conf = float(np.mean(conf))
        
        # Confidence when correct/wrong
        avg_conf_correct = float(np.mean(conf[correct])) if np.sum(correct) > 0 else None
        avg_conf_wrong = float(np.mean(conf[~correct])) if np.sum(~correct) > 0 else None
        
        # Overconfidence rates
        overconf = {}
        for threshold in [0.70, 0.80, 0.90]:
            high_conf = (probs > threshold) | (probs < (1 - threshold))
            if np.sum(high_conf) > 0:
                overconf[f"overconfidence_rate_{int(threshold*100)}"] = float(
                    np.sum(high_conf & ~correct) / np.sum(high_conf)
                )
            else:
                overconf[f"overconfidence_rate_{int(threshold*100)}"] = None
        
        return {
            "avg_confidence": avg_conf,
            "avg_confidence_when_correct": avg_conf_correct,
            "avg_confidence_when_wrong": avg_conf_wrong,
            **overconf,
        }
    
    def _reliability_diagram(
        self,
        probs: np.ndarray,
        actuals: np.ndarray,
    ) -> list:
        """Generate reliability diagram data."""
        n_bins = self.config.confidence_bins
        bins = np.linspace(0, 1, n_bins + 1)
        diagram = []
        
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if np.sum(mask) > 0:
                diagram.append({
                    "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    "avg_confidence": float(np.mean(probs[mask])),
                    "avg_accuracy": float(np.mean(actuals[mask])),
                    "count": int(np.sum(mask)),
                })
        
        return diagram
    
    def _per_category(
        self,
        predictions: list[ModelPrediction],
        questions: list[MarketQuestion],
    ) -> dict:
        """Compute per-category metrics."""
        category_data = defaultdict(lambda: {"correct": 0, "total": 0, "probs": [], "actuals": []})
        
        for pred, q in zip(predictions, questions):
            cat = q.category
            category_data[cat]["total"] += 1
            category_data[cat]["probs"].append(pred.predicted_probability)
            category_data[cat]["actuals"].append(1.0 if q.ground_truth == "yes" else 0.0)
            if pred.predicted_answer == q.ground_truth:
                category_data[cat]["correct"] += 1
        
        result = {}
        for cat, data in category_data.items():
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            probs = np.array(data["probs"])
            actuals = np.array(data["actuals"])
            brier = float(np.mean((probs - actuals) ** 2)) if len(probs) > 0 else None
            
            result[cat] = {
                "accuracy": acc,
                "brier_score": brier,
                "count": data["total"],
            }
        
        return result


# ==============================================================================
# Benchmark Runner
# ==============================================================================

class KalshiBenchRunner:
    """Main benchmark orchestrator."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.data_loader = KalshiDataLoader(config)
        self.metrics_calc = MetricsCalculator(config)
        self.questions: list[MarketQuestion] = []
        self.results: dict[str, EvaluationResult] = {}
        self.dataset_analysis: Optional[DatasetAnalysis] = None
        self.start_time = None
        self.run_timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_data(self):
        """Load benchmark data."""
        self.questions = self.data_loader.load()
        return self
    
    def analyze_dataset(self) -> DatasetAnalysis:
        """Run comprehensive dataset analysis."""
        analyzer = DatasetAnalyzer(self.questions, self.config)
        self.dataset_analysis = analyzer.analyze()
        return self.dataset_analysis
    
    async def evaluate_model(
        self,
        model_config: ModelConfig,
        progress_callback=None,
    ) -> EvaluationResult:
        """Evaluate a single model."""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🤖 EVALUATING: {model_config.name}")
        logger.info("=" * 60)
        logger.info(f"   Model ID:    {model_config.litellm_model}")
        logger.info(f"   Provider:    {model_config.provider.value}")
        logger.info(f"   Cutoff:      {model_config.knowledge_cutoff}")
        if model_config.reasoning_effort:
            logger.info(f"   Reasoning:   {model_config.reasoning_effort}")
        logger.info(f"   Questions:   {len(self.questions):,}")
        logger.info(f"   Concurrency: {self.config.max_concurrent}")
        
        model = LiteLLMModel(model_config)
        predictions = []
        errors = 0
        start_time = datetime.now()
        
        # Run predictions with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Track progress
        completed = 0
        pbar = tqdm(
            total=len(self.questions),
            desc=f"🔮 {model_config.name}",
            unit="q",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        async def predict_with_limit(q: MarketQuestion, idx: int):
            nonlocal completed, errors
            async with semaphore:
                result = await model.predict(q)
                completed += 1
                if result.error:
                    errors += 1
                pbar.update(1)
                pbar.set_postfix({"errors": errors, "ok": completed - errors})
                return result
        
        tasks = [predict_with_limit(q, i) for i, q in enumerate(self.questions)]
        predictions = await asyncio.gather(*tasks)
        pbar.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️  Completed in {elapsed:.1f}s ({len(predictions)/elapsed:.1f} q/s)")
        if errors > 0:
            logger.warning(f"⚠️  {errors} errors during evaluation")
        
        # Compute metrics
        logger.info("📊 Computing metrics...")
        metrics = self.metrics_calc.compute_all(predictions, self.questions)
        
        # Calculate token usage and cost
        total_input_tokens = sum(p.input_tokens for p in predictions)
        total_output_tokens = sum(p.output_tokens for p in predictions)
        total_tokens = total_input_tokens + total_output_tokens
        
        # Calculate cost based on model pricing
        input_cost = (total_input_tokens / 1_000_000) * model_config.price_per_1m_input
        output_cost = (total_output_tokens / 1_000_000) * model_config.price_per_1m_output
        total_cost = input_cost + output_cost
        
        logger.info(f"💰 Token usage: {total_input_tokens:,} in / {total_output_tokens:,} out = {total_tokens:,} total")
        logger.info(f"💵 Cost: ${total_cost:.4f} (${input_cost:.4f} in + ${output_cost:.4f} out)")
        
        # Build result
        result = EvaluationResult(
            model_name=model_config.name,
            model_config=model.get_config(),
            timestamp=datetime.now().isoformat(),
            num_samples=len(self.questions),
            predictions=[asdict(p) for p in predictions],
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 4),
            **metrics,
        )
        
        self.results[model_config.name] = result
        
        # Print summary
        logger.info("")
        logger.info(f"📈 Results for {model_config.name}:")
        logger.info(f"   ┌─────────────────────────────────────┐")
        logger.info(f"   │ Accuracy:      {result.accuracy:>6.2%}              │")
        logger.info(f"   │ Macro F1:      {result.macro_f1:>6.3f}              │")
        logger.info(f"   │ Brier Score:   {result.brier_score:>6.4f}  (↓ better) │")
        logger.info(f"   │ ECE:           {result.ece:>6.4f}  (↓ better) │")
        logger.info(f"   │ Parse Rate:    {result.parse_rate:>6.2%}              │")
        logger.info(f"   └─────────────────────────────────────┘")
        
        return result
    
    def _save_model_result(self, result: EvaluationResult):
        """Save individual model result immediately after evaluation."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        name = result.model_name
        path = os.path.join(
            self.config.output_dir, 
            f"{name.lower().replace(' ', '_').replace('-', '_')}_{self.run_timestamp}.json"
        )
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info(f"💾 Saved: {path}")
    
    def load_existing_results(self) -> int:
        """
        Load existing model result files from the output directory.
        Returns the number of results loaded.
        """
        import glob
        
        if not os.path.isdir(self.config.output_dir):
            return 0
        
        # Find all JSON files that look like model results
        pattern = os.path.join(self.config.output_dir, "*.json")
        files = glob.glob(pattern)
        
        loaded = 0
        for filepath in files:
            filename = os.path.basename(filepath)
            # Skip non-model result files
            if any(x in filename for x in ['summary_', 'metadata_', 'dataset_analysis_', 'run_config']):
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Check if it's a valid model result
                if 'model_name' not in data or 'accuracy' not in data:
                    continue
                
                # Convert dict back to EvaluationResult
                # Handle the nested model_config which may have enum values
                if 'model_config' in data and isinstance(data['model_config'], dict):
                    if 'provider' in data['model_config']:
                        # Convert string back to enum value for display purposes
                        data['model_config']['provider'] = data['model_config']['provider']
                
                result = EvaluationResult(**data)
                self.results[result.model_name] = result
                logger.info(f"📂 Loaded existing result: {result.model_name} (from {filename})")
                loaded += 1
                
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"⚠️  Could not load {filename}: {e}")
                continue
        
        return loaded
    
    async def run(self, model_keys: list[str], load_existing: bool = False) -> dict[str, EvaluationResult]:
        """Run benchmark on specified models."""
        self.start_time = datetime.now()
        
        # Load existing results if requested
        if load_existing:
            logger.info("")
            logger.info("=" * 60)
            logger.info("📂 LOADING EXISTING RESULTS")
            logger.info("=" * 60)
            loaded_count = self.load_existing_results()
            if loaded_count > 0:
                logger.info(f"   Loaded {loaded_count} existing model results")
            else:
                logger.info("   No existing results found")
        
        # Compute knowledge cutoff from models if not explicitly set
        if self.config.knowledge_cutoff is None:
            self.config.knowledge_cutoff = get_max_knowledge_cutoff(model_keys)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("📅 KNOWLEDGE CUTOFF ANALYSIS")
        logger.info("=" * 60)
        logger.info("Model knowledge cutoffs:")
        for key in model_keys:
            if key in MODELS:
                logger.info(f"   {MODELS[key].name:<25} {MODELS[key].knowledge_cutoff}")
        logger.info("")
        logger.info(f"   → Using cutoff: {self.config.knowledge_cutoff} (latest)")
        logger.info(f"   → Only questions resolving AFTER this date")
        logger.info("=" * 60)
        
        self.load_data()
        
        # Run dataset analysis
        self.analyze_dataset()
        
        # Determine which models need to be evaluated (skip those already loaded)
        valid_models = [key for key in model_keys if key in MODELS]
        models_to_run = [key for key in valid_models if MODELS[key].name not in self.results]
        models_skipped = [key for key in valid_models if MODELS[key].name in self.results]
        
        logger.info("")
        if models_skipped:
            logger.info(f"⏭️  Skipping {len(models_skipped)} models with existing results:")
            for key in models_skipped:
                logger.info(f"      {MODELS[key].name}")
        
        if models_to_run:
            logger.info(f"🚀 Starting evaluation of {len(models_to_run)} models")
            logger.info(f"   Total predictions to make: {len(models_to_run) * len(self.questions):,}")
            
            for i, key in enumerate(models_to_run, 1):
                logger.info(f"\n[{i}/{len(models_to_run)}] Starting {MODELS[key].name}...")
                result = await self.evaluate_model(MODELS[key])
                # Save individual result immediately after evaluation
                self._save_model_result(result)
        else:
            logger.info("✅ All models have existing results, nothing to evaluate")
            
        # Final summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏁 BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Total time: {elapsed/60:.1f} minutes")
        logger.info(f"   Models in results: {len(self.results)} ({len(models_skipped)} loaded, {len(models_to_run)} evaluated)")
        logger.info(f"   Questions per model: {len(self.questions):,}")
        
        return self.results
    
    def save_results(self):
        """Save summary results to disk (individual model results already saved during evaluation)."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("💾 SAVING SUMMARY RESULTS")
        logger.info("=" * 60)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = self.run_timestamp  # Use consistent timestamp from run start
        
        # Individual model results already saved during evaluation
        logger.info(f"📄 Individual model results already saved ({len(self.results)} models)")
        
        # 2. Save comparison summary
        summary = self._generate_summary()
        summary_path = os.path.join(self.config.output_dir, f"summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved: {summary_path}")
        
        # 3. Save benchmark metadata
        model_cutoffs = {
            name: MODELS[key].knowledge_cutoff 
            for key in MODELS 
            for name, result in self.results.items() 
            if MODELS[key].name == name
        }
        
        metadata = {
            "benchmark": "KalshiBench",
            "version": "1.0",
            "timestamp": timestamp,
            "config": asdict(self.config),
            "num_questions": len(self.questions),
            "knowledge_cutoff_used": self.config.knowledge_cutoff,
            "model_knowledge_cutoffs": {
                result.model_name: result.model_config.get("knowledge_cutoff", "unknown")
                for result in self.results.values()
            },
            "categories": list(set(q.category for q in self.questions)),
            "date_range": {
                "earliest": min(q.close_time for q in self.questions),
                "latest": max(q.close_time for q in self.questions),
            },
            "ground_truth_distribution": {
                "yes": sum(1 for q in self.questions if q.ground_truth == "yes"),
                "no": sum(1 for q in self.questions if q.ground_truth == "no"),
            },
        }
        meta_path = os.path.join(self.config.output_dir, f"metadata_{timestamp}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved: {meta_path}")
        
        # 4. Save dataset analysis
        if self.dataset_analysis:
            analysis_path = os.path.join(self.config.output_dir, f"dataset_analysis_{timestamp}.json")
            with open(analysis_path, "w") as f:
                json.dump(self.dataset_analysis.model_dump(), f, indent=2, default=str)
            print(f"Saved: {analysis_path}")
        
        # 5. Generate paper prompts
        self._save_paper_prompts(timestamp)
        
        # 6. Generate markdown report
        self._save_markdown_report(timestamp)
        
        return summary_path
    
    def _generate_summary(self) -> dict:
        """Generate comparison summary across all models."""
        summary = {
            "benchmark": "KalshiBench",
            "timestamp": datetime.now().isoformat(),
            "knowledge_cutoff_used": self.config.knowledge_cutoff,
            "num_models": len(self.results),
            "num_samples": len(self.questions),
            "models": {},
            "leaderboard": {
                "by_accuracy": [],
                "by_brier_score": [],
                "by_calibration": [],
            },
        }
        
        for name, result in self.results.items():
            summary["models"][name] = {
                "knowledge_cutoff": result.model_config.get("knowledge_cutoff", "unknown"),
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "brier_score": result.brier_score,
                "brier_skill_score": result.brier_skill_score,
                "ece": result.ece,
                "mce": result.mce,
                "log_loss": result.log_loss,
                "avg_confidence": result.avg_confidence,
                "overconfidence_rate_80": result.overconfidence_rate_80,
                "parse_rate": result.parse_rate,
                "avg_latency_ms": result.avg_latency_ms,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "total_tokens": result.total_tokens,
                "total_cost_usd": result.total_cost_usd,
            }
        
        # Leaderboards
        models_list = list(summary["models"].items())
        
        summary["leaderboard"]["by_accuracy"] = sorted(
            [{"model": k, "accuracy": v["accuracy"]} for k, v in models_list],
            key=lambda x: x["accuracy"],
            reverse=True,
        )
        
        summary["leaderboard"]["by_brier_score"] = sorted(
            [{"model": k, "brier_score": v["brier_score"]} for k, v in models_list if v["brier_score"]],
            key=lambda x: x["brier_score"],  # Lower is better
        )
        
        summary["leaderboard"]["by_calibration"] = sorted(
            [{"model": k, "ece": v["ece"]} for k, v in models_list if v["ece"]],
            key=lambda x: x["ece"],  # Lower is better
        )
        
        return summary
    
    def _save_paper_prompts(self, timestamp: str):
        """Save prompts for generating paper sections."""
        summary = self._generate_summary()
        da = self.dataset_analysis  # Shorthand
        
        # Build dataset analysis section for prompts
        dataset_analysis_text = ""
        if da:
            # Helper for time horizon stats (may be None)
            th = da.time_horizon_stats
            th_mean = th.mean_days_ahead if th else 0
            th_median = th.median_days_ahead if th else 0
            th_min = th.min_days_ahead if th else 0
            th_max = th.max_days_ahead if th else 0
            
            dataset_analysis_text = f"""
## DATASET ANALYSIS (Critical for Paper)

### Temporal Characteristics
- Date Range: {da.date_range.earliest or 'N/A'} to {da.date_range.latest or 'N/A'}
- Temporal Span: {da.date_range.span_days} days
- Time Horizon (days from cutoff to resolution):
  - Mean: {th_mean:.1f} days
  - Median: {th_median:.1f} days
  - Range: {th_min} - {th_max} days

### Ground Truth Analysis
- Overall Yes Rate: {da.ground_truth_distribution.yes_rate:.1%}
- Yes: {da.ground_truth_distribution.yes}, No: {da.ground_truth_distribution.no}
- Ground Truth by Category:
{json.dumps({k: v.model_dump() for k, v in da.ground_truth_by_category.items()}, indent=2)}

### Category Distribution
- Number of Categories: {da.num_categories}
- Category Entropy: {da.category_entropy:.2f} bits (measures diversity)
- Largest Category: {da.largest_category}
- Smallest Category: {da.smallest_category}
- Full Distribution:
{json.dumps(da.category_distribution, indent=2)}

### Question Complexity
- Question Length: mean={da.question_length_stats.mean:.0f}, median={da.question_length_stats.median:.0f}, std={da.question_length_stats.std:.0f} chars
- Description Length: mean={da.description_length_stats.mean:.0f}, median={da.description_length_stats.median:.0f} chars
- Vocabulary Size: {da.total_vocabulary_size:,} unique words
- Avg Words per Question: {da.avg_words_per_question:.1f}
"""
        
        # Methods section prompt
        methods_prompt = f"""You are a scientific writer preparing the Methods section for a NeurIPS Datasets & Benchmarks paper.

## Benchmark: KalshiBench
{dataset_analysis_text}

### Data Source
- Platform: Kalshi (CFTC-regulated prediction market)
- Dataset: {self.config.dataset_name}
- Total questions evaluated: {len(self.questions)}

### Temporal Filtering
- Knowledge cutoff: {self.config.knowledge_cutoff}
- All questions resolve AFTER this date to prevent memorization
- This is critical for fair evaluation of models with different training cutoffs

### Categories Present
{json.dumps(list(set(q.category for q in self.questions)), indent=2)}

### Ground Truth Distribution
- Yes: {sum(1 for q in self.questions if q.ground_truth == "yes")}
- No: {sum(1 for q in self.questions if q.ground_truth == "no")}

### Evaluation Protocol
1. Each model receives the question text and description
2. Models must provide:
   - Binary prediction (yes/no)
   - Confidence score (0-100%)
   - Reasoning
3. Responses parsed using XML tags: <think>, <answer>, <confidence>

### Metrics
**Classification:**
- Accuracy, Precision, Recall, F1 (per-class and macro)

**Calibration:**
- Brier Score: Mean squared error between predicted probability and outcome
- Brier Skill Score: Improvement over climatological baseline
- ECE (Expected Calibration Error): Weighted average of |confidence - accuracy| per bin
- MCE (Maximum Calibration Error): Worst-calibrated bin
- ACE (Adaptive Calibration Error): ECE with equal-mass bins
- Log Loss: Cross-entropy loss (penalizes confident wrong predictions)

**Confidence Analysis:**
- Overconfidence rates at 70%, 80%, 90% thresholds
- Average confidence when correct vs wrong

### Models Evaluated
{json.dumps([{"name": k, "provider": v["provider"]} for k, v in [(name, MODELS[name.lower().replace('-', '_').replace('.', '_').replace(' ', '-')].model_dump(mode="json")) for name in self.results.keys() if name.lower().replace('-', '_').replace('.', '_').replace(' ', '-') in MODELS] if v], indent=2, default=str)}

Write a formal Methods section (3-4 paragraphs) suitable for NeurIPS covering:
1. Benchmark construction and rationale (USE THE DATASET ANALYSIS STATS!)
2. Evaluation protocol and metrics
3. Model selection criteria
4. Description of the dataset characteristics (temporal span, category diversity, question complexity)
"""
        
        # Results section prompt
        results_data = {name: {
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "brier_score": r.brier_score,
            "brier_skill_score": r.brier_skill_score,
            "ece": r.ece,
            "mce": r.mce,
            "log_loss": r.log_loss,
            "overconfidence_rate_80": r.overconfidence_rate_80,
            "per_category": r.per_category,
            "reliability_diagram": r.reliability_diagram,
        } for name, r in self.results.items()}
        
        results_prompt = f"""You are a scientific writer preparing the Results section for a NeurIPS Datasets & Benchmarks paper on KalshiBench.
{dataset_analysis_text}

## Evaluation Results

### Model Performance Summary
{json.dumps(results_data, indent=2, default=str)}

### Leaderboards
{json.dumps(summary["leaderboard"], indent=2)}

## Writing Instructions

Structure the Results section as follows:

**Paragraph 1: Main Forecasting Performance**
- Report accuracy and F1 across all models
- Identify best and worst performers
- Note any surprising results (e.g., smaller models outperforming larger ones)
- Highlight any model family patterns (e.g., reasoning models vs standard models)

**Paragraph 2: Calibration Analysis (KEY CONTRIBUTION)**
- This is the main contribution of KalshiBench
- Compare Brier Scores across models (THE key metric for forecasting)
- Discuss ECE - which models are best calibrated?
- Analyze gap between accuracy and calibration (a model can be accurate but poorly calibrated)
- Compare Brier Skill Score to show improvement over naive baseline

**Paragraph 3: Overconfidence Analysis**
- Report overconfidence rates at 80% threshold
- Which models are most/least overconfident?
- Discuss implications for real-world deployment
- Do reasoning models (o1, DeepSeek-R1, QwQ) show better calibration?

**Paragraph 4: Category Breakdown**
- Are there categories where models struggle?
- Do different models have different strengths?
- Which categories show best/worst calibration?

**Paragraph 5: Key Findings**
- Summarize main takeaways
- What does this reveal about current LLM forecasting capabilities?
- Implications for using LLMs for forecasting tasks
- Recommendations for model selection based on calibration needs

Use precise numbers, scientific language, and draw meaningful conclusions. Create 1-2 tables summarizing key results.
"""
        
        # Dataset section prompt (dedicated for NeurIPS Datasets & Benchmarks)
        dataset_prompt = ""
        if da:
            # Helper for time horizon stats (may be None)
            th = da.time_horizon_stats
            th_mean = th.mean_days_ahead if th else 0
            th_median = th.median_days_ahead if th else 0
            th_min = th.min_days_ahead if th else 0
            th_max = th.max_days_ahead if th else 0
            
            # Calculate imbalance ratio
            yes_ct = da.ground_truth_distribution.yes or 1
            no_ct = da.ground_truth_distribution.no or 1
            imbalance = max(yes_ct, no_ct) / max(min(yes_ct, no_ct), 1)
            
            dataset_prompt = f"""You are a scientific writer preparing the Dataset section for a NeurIPS Datasets & Benchmarks paper.

## Dataset: KalshiBench

### What is KalshiBench?

KalshiBench is a benchmark for evaluating LLM forecasting calibration using real-world prediction markets from Kalshi, a CFTC-regulated prediction market platform. Unlike traditional benchmarks that measure accuracy on static knowledge, KalshiBench evaluates whether models can make well-calibrated probabilistic predictions on questions with verifiable real-world outcomes.

**Key Innovation:** By using temporally-filtered prediction market questions, we ensure models cannot have memorized the answers during training—the questions resolve AFTER the models' knowledge cutoffs.

### Overview Statistics
- Total Questions: {da.total_questions}
- Knowledge Cutoff: {da.knowledge_cutoff}
- Source: Kalshi (CFTC-regulated prediction market)

### Temporal Characteristics
- Date Range: {da.date_range.earliest or 'N/A'} to {da.date_range.latest or 'N/A'}
- Temporal Span: {da.date_range.span_days} days
- Mean Time Horizon: {th_mean:.1f} days (from cutoff to resolution)
- Median Time Horizon: {th_median:.1f} days
- Time Horizon Range: {th_min} - {th_max} days

### Label Distribution Analysis
- Yes Rate: {da.ground_truth_distribution.yes_rate:.1%}
- Yes Count: {da.ground_truth_distribution.yes}
- No Count: {da.ground_truth_distribution.no}
- Imbalance Ratio: {imbalance:.2f}:1

### Ground Truth by Category (potential biases!)
{json.dumps({k: v.model_dump() for k, v in da.ground_truth_by_category.items()}, indent=2)}

### Ground Truth by Month (temporal bias analysis)
{json.dumps({k: v.model_dump() for k, v in da.ground_truth_by_month.items()}, indent=2)}

### Category Distribution
- Number of Categories: {da.num_categories}
- Shannon Entropy: {da.category_entropy:.2f} bits (higher = more diverse)
- Largest Category: {da.largest_category}
- Smallest Category: {da.smallest_category}
- Distribution:
{json.dumps(da.category_distribution, indent=2)}

### Question Complexity Analysis
- Question Length: mean={da.question_length_stats.mean:.0f}, median={da.question_length_stats.median:.0f}, std={da.question_length_stats.std:.0f} characters
- Description Length: mean={da.description_length_stats.mean:.0f}, median={da.description_length_stats.median:.0f} characters  
- Total Vocabulary: {da.total_vocabulary_size:,} unique words
- Avg Words per Question: {da.avg_words_per_question:.1f}

## Writing Instructions

Write a formal Dataset section (2-3 paragraphs) suitable for NeurIPS Datasets & Benchmarks covering:

**Paragraph 1: Data Collection & Source**
- Describe Kalshi as the data source (CFTC-regulated, real money)
- Explain why prediction markets are valuable for forecasting evaluation
- Questions have real-world outcomes that can be objectively verified
- Mention temporal filtering to prevent data contamination

**Paragraph 2: Dataset Construction & Processing**
- Describe the two-step dataset creation process:
  1. Raw data extraction from Kalshi API
  2. Cleaning, deduplication, and standardization into KalshiBench dataset
- Explain that dataset is deduplicated by series_ticker to avoid related questions
- Note that only resolved binary yes/no markets are included

**Paragraph 3: Dataset Statistics & Characteristics**
- Report key statistics (size, categories, temporal span)
- Discuss label distribution and potential biases
- Describe question complexity metrics
- Note the category diversity (entropy measure)

**Key Points to Emphasize:**
1. Real-world grounding: Questions have real monetary stakes
2. Temporal integrity: Knowledge cutoff prevents memorization
3. Clean evaluation: Pre-processed, deduplicated dataset
4. Diversity: Multiple categories spanning politics, economics, science, etc.

Create a summary statistics table with the most important metrics.
"""
        
        # Save prompts
        methods_path = os.path.join(self.config.output_dir, f"paper_methods_prompt_{timestamp}.txt")
        results_path = os.path.join(self.config.output_dir, f"paper_results_prompt_{timestamp}.txt")
        dataset_path = os.path.join(self.config.output_dir, f"paper_dataset_prompt_{timestamp}.txt")
        
        with open(methods_path, "w") as f:
            f.write(methods_prompt)
        print(f"Saved: {methods_path}")
        
        with open(results_path, "w") as f:
            f.write(results_prompt)
        print(f"Saved: {results_path}")
        
        if dataset_prompt:
            with open(dataset_path, "w") as f:
                f.write(dataset_prompt)
            print(f"Saved: {dataset_path}")
    
    def _save_markdown_report(self, timestamp: str):
        """Save a comprehensive human-readable markdown report."""
        summary = self._generate_summary()
        da = self.dataset_analysis
        
        # =====================================================================
        # DATASET ANALYSIS SECTION
        # =====================================================================
        dataset_section = ""
        if da:
            th = da.time_horizon_stats
            th_mean = th.mean_days_ahead if th else 0
            th_median = th.median_days_ahead if th else 0
            th_std = th.std_days_ahead if th else 0
            th_min = th.min_days_ahead if th else 0
            th_max = th.max_days_ahead if th else 0
            
            # Category distribution table
            cat_table = "| Category | Count | % | Yes Rate |\n"
            cat_table += "|----------|-------|---|----------|\n"
            for cat, count in sorted(da.category_distribution.items(), key=lambda x: -x[1]):
                pct = count / da.total_questions * 100
                yes_rate = da.ground_truth_by_category.get(cat)
                yr_str = f"{yes_rate.yes_rate:.1%}" if yes_rate else "N/A"
                cat_table += f"| {cat} | {count} | {pct:.1f}% | {yr_str} |\n"
            
            # Monthly distribution table
            month_table = "| Month | Count | Yes Rate |\n"
            month_table += "|-------|-------|----------|\n"
            for month, data in sorted(da.ground_truth_by_month.items()):
                month_table += f"| {month} | {data.total} | {data.yes_rate:.1%} |\n"
            
            dataset_section = f"""
## Dataset Analysis

### Overview
- **Total Questions:** {da.total_questions}
- **Knowledge Cutoff:** {da.knowledge_cutoff}
- **Source:** Kalshi (CFTC-regulated prediction market)

### Temporal Distribution

| Metric | Value |
|--------|-------|
| Date Range | {da.date_range.earliest or 'N/A'} to {da.date_range.latest or 'N/A'} |
| Temporal Span | {da.date_range.span_days} days |
| Mean Time Horizon | {th_mean:.1f} days |
| Median Time Horizon | {th_median:.1f} days |
| Std Dev Time Horizon | {th_std:.1f} days |
| Min Time Horizon | {th_min} days |
| Max Time Horizon | {th_max} days |

#### Questions by Month

{month_table}

### Ground Truth Distribution

| Outcome | Count | Percentage |
|---------|-------|------------|
| Yes | {da.ground_truth_distribution.yes} | {da.ground_truth_distribution.yes_rate:.1%} |
| No | {da.ground_truth_distribution.no} | {1 - da.ground_truth_distribution.yes_rate:.1%} |

### Category Distribution

- **Number of Categories:** {da.num_categories}
- **Category Entropy:** {da.category_entropy:.2f} bits (max = {np.log2(da.num_categories):.2f} bits)
- **Largest Category:** {da.largest_category} ({da.category_distribution.get(da.largest_category, 0)} questions)
- **Smallest Category:** {da.smallest_category} ({da.category_distribution.get(da.smallest_category, 0)} questions)

{cat_table}

### Question Complexity

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
| Question Length (chars) | {da.question_length_stats.mean:.0f} | {da.question_length_stats.median:.0f} | {da.question_length_stats.std:.0f} | {da.question_length_stats.min} | {da.question_length_stats.max} |
| Description Length (chars) | {da.description_length_stats.mean:.0f} | {da.description_length_stats.median:.0f} | {da.description_length_stats.std:.0f} | {da.description_length_stats.min} | {da.description_length_stats.max} |

- **Total Vocabulary Size:** {da.total_vocabulary_size:,} unique words
- **Average Words per Question:** {da.avg_words_per_question:.1f}

"""
        
        # =====================================================================
        # SUMMARY TABLES
        # =====================================================================
        
        # Classification summary table
        accuracy_table = "| Model | Accuracy | Macro F1 | Precision (Yes) | Recall (Yes) | F1 (Yes) | Parse Rate |\n"
        accuracy_table += "|-------|----------|----------|-----------------|--------------|----------|------------|\n"
        for name, result in self.results.items():
            accuracy_table += f"| {name} | {result.accuracy:.2%} | {result.macro_f1:.3f} | {result.precision_yes:.3f} | {result.recall_yes:.3f} | {result.f1_yes:.3f} | {result.parse_rate:.2%} |\n"
        
        # Calibration summary table
        calibration_table = "| Model | Brier | BSS | ECE | MCE | ACE | Log Loss |\n"
        calibration_table += "|-------|-------|-----|-----|-----|-----|----------|\n"
        for name, result in self.results.items():
            brier = f"{result.brier_score:.4f}" if result.brier_score else "N/A"
            bss = f"{result.brier_skill_score:.4f}" if result.brier_skill_score else "N/A"
            ece = f"{result.ece:.4f}" if result.ece else "N/A"
            mce = f"{result.mce:.4f}" if result.mce else "N/A"
            ace = f"{result.ace:.4f}" if result.ace else "N/A"
            ll = f"{result.log_loss:.4f}" if result.log_loss else "N/A"
            calibration_table += f"| {name} | {brier} | {bss} | {ece} | {mce} | {ace} | {ll} |\n"
        
        # Confidence analysis table
        confidence_table = "| Model | Avg Conf | Conf When Correct | Conf When Wrong | Overconf@70% | Overconf@80% | Overconf@90% |\n"
        confidence_table += "|-------|----------|-------------------|-----------------|--------------|--------------|---------------|\n"
        for name, result in self.results.items():
            avg_conf = f"{result.avg_confidence:.2%}" if result.avg_confidence else "N/A"
            conf_correct = f"{result.avg_confidence_when_correct:.2%}" if result.avg_confidence_when_correct else "N/A"
            conf_wrong = f"{result.avg_confidence_when_wrong:.2%}" if result.avg_confidence_when_wrong else "N/A"
            oc70 = f"{result.overconfidence_rate_70:.2%}" if result.overconfidence_rate_70 else "N/A"
            oc80 = f"{result.overconfidence_rate_80:.2%}" if result.overconfidence_rate_80 else "N/A"
            oc90 = f"{result.overconfidence_rate_90:.2%}" if result.overconfidence_rate_90 else "N/A"
            confidence_table += f"| {name} | {avg_conf} | {conf_correct} | {conf_wrong} | {oc70} | {oc80} | {oc90} |\n"
        
        # Token usage and cost table
        cost_table = "| Model | Input Tokens | Output Tokens | Total Tokens | Cost (USD) |\n"
        cost_table += "|-------|--------------|---------------|--------------|------------|\n"
        total_all_tokens = 0
        total_all_cost = 0.0
        for name, result in self.results.items():
            cost_table += f"| {name} | {result.total_input_tokens:,} | {result.total_output_tokens:,} | {result.total_tokens:,} | ${result.total_cost_usd:.4f} |\n"
            total_all_tokens += result.total_tokens
            total_all_cost += result.total_cost_usd
        cost_table += f"| **TOTAL** | - | - | **{total_all_tokens:,}** | **${total_all_cost:.4f}** |\n"
        
        # =====================================================================
        # PER-MODEL DETAILED SECTIONS
        # =====================================================================
        model_details = ""
        for name, result in self.results.items():
            # Confusion matrix
            cm = result.confusion_matrix
            cm_table = f"""
```
                 Predicted
              Yes        No
           ┌─────────┬─────────┐
Actual Yes │   {cm['tp']:>4}   │   {cm['fn']:>4}   │
           ├─────────┼─────────┤
Actual No  │   {cm['fp']:>4}   │   {cm['tn']:>4}   │
           └─────────┴─────────┘
```
"""
            
            # Reliability diagram as ASCII
            rel_diagram = "| Bin | Avg Confidence | Avg Accuracy | Count | Gap |\n"
            rel_diagram += "|-----|----------------|--------------|-------|-----|\n"
            for bin_data in result.reliability_diagram:
                gap = bin_data['avg_confidence'] - bin_data['avg_accuracy']
                gap_str = f"+{gap:.3f}" if gap > 0 else f"{gap:.3f}"
                rel_diagram += f"| {bin_data['bin']} | {bin_data['avg_confidence']:.3f} | {bin_data['avg_accuracy']:.3f} | {bin_data['count']} | {gap_str} |\n"
            
            # Category breakdown
            cat_breakdown = "| Category | Accuracy | Brier Score | Count |\n"
            cat_breakdown += "|----------|----------|-------------|-------|\n"
            for cat, data in sorted(result.per_category.items(), key=lambda x: -x[1]['count']):
                brier = f"{data['brier_score']:.4f}" if data['brier_score'] else "N/A"
                cat_breakdown += f"| {cat} | {data['accuracy']:.2%} | {brier} | {data['count']} |\n"
            
            reasoning_str = f"\n- **Reasoning Effort:** {result.model_config.get('reasoning_effort')}" if result.model_config.get('reasoning_effort') else ""
            model_details += f"""
---

## {name}

**Model Configuration:**
- **Provider:** {result.model_config.get('provider', 'N/A')}
- **Model ID:** {result.model_config.get('litellm_model', 'N/A')}
- **Knowledge Cutoff:** {result.model_config.get('knowledge_cutoff', 'N/A')}
- **Temperature:** {result.model_config.get('temperature', 'N/A')}{reasoning_str}

### All Metrics

#### Classification Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {result.accuracy:.4f} ({result.accuracy:.2%}) |
| Macro F1 | {result.macro_f1:.4f} |
| Precision (Yes) | {result.precision_yes:.4f} |
| Recall (Yes) | {result.recall_yes:.4f} |
| F1 (Yes) | {result.f1_yes:.4f} |
| Precision (No) | {result.precision_no:.4f} |
| Recall (No) | {result.recall_no:.4f} |
| F1 (No) | {result.f1_no:.4f} |

#### Calibration Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Brier Score | {result.brier_score:.4f} | Lower is better (0 = perfect) |
| Brier Skill Score | {result.brier_skill_score:.4f} | Higher is better (improvement over base rate) |
| ECE | {result.ece:.4f} | Lower is better (expected calibration error) |
| MCE | {result.mce:.4f} | Lower is better (max calibration error) |
| ACE | {result.ace:.4f} | Lower is better (adaptive calibration error) |
| Log Loss | {result.log_loss:.4f} | Lower is better |

#### Confidence Analysis

| Metric | Value |
|--------|-------|
| Average Confidence | {result.avg_confidence:.4f} |
| Avg Confidence When Correct | {result.avg_confidence_when_correct if result.avg_confidence_when_correct else 'N/A'} |
| Avg Confidence When Wrong | {result.avg_confidence_when_wrong if result.avg_confidence_when_wrong else 'N/A'} |
| Overconfidence Rate @70% | {result.overconfidence_rate_70 if result.overconfidence_rate_70 else 'N/A'} |
| Overconfidence Rate @80% | {result.overconfidence_rate_80 if result.overconfidence_rate_80 else 'N/A'} |
| Overconfidence Rate @90% | {result.overconfidence_rate_90 if result.overconfidence_rate_90 else 'N/A'} |

#### Performance Metrics

| Metric | Value |
|--------|-------|
| Parse Rate | {result.parse_rate:.4f} ({result.parse_rate:.2%}) |
| Avg Latency | {result.avg_latency_ms:.1f} ms |

#### Token Usage & Cost

| Metric | Value |
|--------|-------|
| Input Tokens | {result.total_input_tokens:,} |
| Output Tokens | {result.total_output_tokens:,} |
| Total Tokens | {result.total_tokens:,} |
| Total Cost | ${result.total_cost_usd:.4f} |

### Confusion Matrix

{cm_table}

- **True Positives (TP):** {cm['tp']} - Correctly predicted "yes"
- **True Negatives (TN):** {cm['tn']} - Correctly predicted "no"
- **False Positives (FP):** {cm['fp']} - Incorrectly predicted "yes" (actual was "no")
- **False Negatives (FN):** {cm['fn']} - Incorrectly predicted "no" (actual was "yes")

### Reliability Diagram

Shows calibration: ideally, avg_accuracy should equal avg_confidence in each bin.

{rel_diagram}

**Interpretation:**
- **Gap > 0:** Model is overconfident (confidence > accuracy)
- **Gap < 0:** Model is underconfident (confidence < accuracy)
- **Gap ≈ 0:** Well-calibrated

### Category Breakdown

{cat_breakdown}

"""
        
        # =====================================================================
        # LEADERBOARDS
        # =====================================================================
        leaderboard_section = f"""
## Leaderboards

### By Accuracy (Higher is Better)

| Rank | Model | Accuracy |
|------|-------|----------|
{chr(10).join([f"| {i+1} | {m['model']} | {m['accuracy']:.2%} |" for i, m in enumerate(summary['leaderboard']['by_accuracy'])])}

### By Brier Score (Lower is Better)

| Rank | Model | Brier Score |
|------|-------|-------------|
{chr(10).join([f"| {i+1} | {m['model']} | {m['brier_score']:.4f} |" for i, m in enumerate(summary['leaderboard']['by_brier_score'])])}

### By ECE (Lower is Better)

| Rank | Model | ECE |
|------|-------|-----|
{chr(10).join([f"| {i+1} | {m['model']} | {m['ece']:.4f} |" for i, m in enumerate(summary['leaderboard']['by_calibration'])])}

"""
        
        # =====================================================================
        # ASSEMBLE FULL REPORT
        # =====================================================================
        report = f"""# KalshiBench Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Benchmark Version:** 1.0

---

## Executive Summary

KalshiBench evaluates language model forecasting ability and calibration using temporally-filtered prediction market questions from Kalshi, a CFTC-regulated prediction market.

| Metric | Value |
|--------|-------|
| Total Questions | {len(self.questions)} |
| Knowledge Cutoff | {self.config.knowledge_cutoff} |
| Models Evaluated | {len(self.results)} |
| Date Range | {min(q.close_time for q in self.questions)} to {max(q.close_time for q in self.questions)} |
| Yes Rate | {sum(1 for q in self.questions if q.ground_truth == "yes")/len(self.questions):.1%} |
| Random Seed | {self.config.seed} |

{dataset_section}

---

## Model Comparison Summary

### Classification Performance

{accuracy_table}

### Calibration Metrics

{calibration_table}

**Metric Definitions:**
- **Brier:** Brier Score - mean squared error of probability predictions (lower = better)
- **BSS:** Brier Skill Score - improvement over always predicting base rate (higher = better)
- **ECE:** Expected Calibration Error - avg |confidence - accuracy| weighted by bin size (lower = better)
- **MCE:** Maximum Calibration Error - worst-calibrated bin (lower = better)
- **ACE:** Adaptive Calibration Error - ECE with equal-mass bins (lower = better)

### Confidence Analysis

{confidence_table}

**Overconfidence Rate @X%:** Fraction of wrong predictions among those with confidence > X%

### Token Usage & Cost

{cost_table}

{leaderboard_section}

---

# Detailed Model Results

{model_details}

---

## Files Generated

| File | Description |
|------|-------------|
| `summary_*.json` | Aggregated results for all models |
| `<model>_*.json` | Individual model results with all predictions |
| `metadata_*.json` | Benchmark configuration and dataset statistics |
| `dataset_analysis_*.json` | Comprehensive dataset analysis |
| `paper_methods_prompt_*.txt` | Prompt for generating Methods section |
| `paper_results_prompt_*.txt` | Prompt for generating Results section |
| `paper_dataset_prompt_*.txt` | Prompt for generating Dataset section |
| `report_*.md` | This report |

---

## Citation

```bibtex
@misc{{kalshibench2024,
  title={{KalshiBench: Evaluating LLM Forecasting Calibration via Prediction Markets}},
  author={{2084 Collective}},
  year={{2024}},
  note={{Evaluation of {len(self.results)} models on {len(self.questions)} prediction market questions}}
}}
```

---

*Report generated by KalshiBench v1.0*
"""
        
        report_path = os.path.join(self.config.output_dir, f"report_{timestamp}.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Saved: {report_path}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KalshiBench: Evaluate LLM forecasting calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset:
  By default, loads from pre-cleaned '2084Collective/kalshibench-v2' dataset.
  Use --raw to load from raw dataset with on-the-fly deduplication.

Available models:
  OpenAI:    gpt-5.2, gpt-5.2-low, gpt-5.2-high, gpt-5.2-xhigh,
             gpt-5.1, gpt-5.1-medium, gpt-5.1-high,
             gpt-5-mini, gpt-5-nano,
             gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini
  Anthropic: claude-opus-4.5, claude-sonnet-4.5, claude-3-5-sonnet, claude-3-5-haiku
  Google:    gemini-3-pro, gemini-3-flash, gemini-2.5-pro, gemini-2.0-flash
  Together:  qwen3-235b, qwen3-235b-thinking, qwen3-32b, qwen-2.5-72b, qwen-2.5-7b, qwen-qwq-32b,
             llama-3.3-70b, llama-3.1-405b, llama-3.1-8b, mistral-large
  Fireworks: deepseek-v3, deepseek-v3.1, deepseek-v3.2, deepseek-r1, kimi-k2

Reasoning effort (GPT-5.x only): low, medium, high, xhigh (5.2 only)

Note: Knowledge cutoff is automatically computed as the LATEST cutoff among
      all selected models, ensuring fair evaluation on unseen data.

Example:
  python kalshibench.py --models gpt-5.2 claude-opus-4.5 --samples 100
        """,
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["gpt-4o-mini"],
        help="Models to evaluate (space-separated)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=200,
        help="Number of questions to evaluate (default: 200)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="kalshibench_results",
        help="Output directory (default: kalshibench_results)",
    )
    parser.add_argument(
        "--cutoff", "-c",
        type=str,
        default=None,
        help="Knowledge cutoff date YYYY-MM-DD (default: auto-computed from models)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw dataset with on-the-fly deduplication (legacy mode)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name (default: 2084Collective/kalshibench-v2)",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing results from output directory (skip models already evaluated)",
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 80)
        print(f"  {'Key':<20} {'Name':<25} {'Provider':<12} {'Cutoff':<12}")
        print("-" * 80)
        for key, config in MODELS.items():
            print(f"  {key:<20} {config.name:<25} {config.provider.value:<12} {config.knowledge_cutoff}")
        return
    
    # Validate models
    invalid = [m for m in args.models if m not in MODELS]
    if invalid:
        print(f"Error: Unknown models: {invalid}")
        print(f"Use --list-models to see available options")
        return
    
    # Create config
    config = BenchmarkConfig(
        num_samples=args.samples,
        output_dir=args.output,
        knowledge_cutoff=args.cutoff,
        max_concurrent=args.concurrent,
        use_raw_dataset=args.raw,
    )
    
    # Override dataset name if provided
    if args.dataset:
        if args.raw:
            config.raw_dataset_name = args.dataset
        else:
            config.dataset_name = args.dataset
    
    print("\n" + "=" * 60)
    print("KalshiBench - LLM Forecasting Calibration Benchmark")
    print("=" * 60)
    dataset_mode = "raw (with dedup)" if args.raw else "pre-cleaned"
    dataset_name = config.raw_dataset_name if args.raw else config.dataset_name
    print(f"Dataset: {dataset_name} ({dataset_mode})")
    print(f"Models: {', '.join(args.models)}")
    print(f"Samples: {args.samples}")
    print(f"Knowledge Cutoff: {args.cutoff or 'auto-computed'}")
    print(f"Output: {args.output}")
    
    # Run benchmark
    runner = KalshiBenchRunner(config)
    asyncio.run(runner.run(args.models, load_existing=args.load_existing))
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    runner.save_results()
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

