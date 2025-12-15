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


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    litellm_model: str
    provider: ModelProvider
    knowledge_cutoff: str  # YYYY-MM-DD format
    description: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    # Pricing per 1M tokens (input, output) in USD
    price_per_1m_input: float = 0.0
    price_per_1m_output: float = 0.0


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
        litellm_model="gpt-5.2",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-10-01",
        description="OpenAI GPT-5.2 (Dec 2025)",
        price_per_1m_input=5.00,
        price_per_1m_output=15.00,
    ),
    "gpt-5.1": ModelConfig(
        name="GPT-5.1",
        litellm_model="gpt-5.1",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2025-08-01",
        description="OpenAI GPT-5.1 (Nov 2025)",
        price_per_1m_input=4.00,
        price_per_1m_output=12.00,
    ),
    # OpenAI - GPT-4o series
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        litellm_model="gpt-4o",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI GPT-4o flagship multimodal",
        price_per_1m_input=2.50,
        price_per_1m_output=10.00,
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        litellm_model="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI GPT-4o-mini (fast, affordable)",
        price_per_1m_input=0.15,
        price_per_1m_output=0.60,
    ),
    # OpenAI - o1/o3 reasoning series
    "o1": ModelConfig(
        name="o1",
        litellm_model="o1",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI o1 reasoning model",
        temperature=1.0,
        price_per_1m_input=15.00,
        price_per_1m_output=60.00,
    ),
    "o1-mini": ModelConfig(
        name="o1-mini",
        litellm_model="o1-mini",
        provider=ModelProvider.OPENAI,
        knowledge_cutoff="2024-10-01",
        description="OpenAI o1-mini (faster reasoning)",
        temperature=1.0,
        price_per_1m_input=3.00,
        price_per_1m_output=12.00,
    ),
    "o3-mini": ModelConfig(
        name="o3-mini",
        litellm_model="o3-mini",
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
        litellm_model="claude-opus-4-5-20241120",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2025-04-01",
        description="Anthropic Claude Opus 4.5 (Nov 2025)",
        price_per_1m_input=15.00,
        price_per_1m_output=75.00,
    ),
    "claude-sonnet-4.5": ModelConfig(
        name="Claude-Sonnet-4.5",
        litellm_model="claude-sonnet-4-5-20250929",
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
    "claude-3-5-haiku": ModelConfig(
        name="Claude-3.5-Haiku",
        litellm_model="claude-3-5-haiku-latest",
        provider=ModelProvider.ANTHROPIC,
        knowledge_cutoff="2024-04-01",
        description="Anthropic Claude 3.5 Haiku (fast)",
        price_per_1m_input=0.80,
        price_per_1m_output=4.00,
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
    # Together AI - DeepSeek
    # ==========================================================================
    "deepseek-v3": ModelConfig(
        name="DeepSeek-V3",
        litellm_model="together_ai/deepseek-ai/DeepSeek-V3",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-11-01",
        description="DeepSeek V3 via Together AI",
        price_per_1m_input=0.90,
        price_per_1m_output=0.90,
    ),
    "deepseek-r1": ModelConfig(
        name="DeepSeek-R1",
        litellm_model="together_ai/deepseek-ai/DeepSeek-R1",
        provider=ModelProvider.TOGETHER,
        knowledge_cutoff="2024-11-01",
        description="DeepSeek R1 reasoning model via Together AI",
        price_per_1m_input=3.00,
        price_per_1m_output=7.00,
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
    "gemini-2.0-flash": ModelConfig(
        name="Gemini-2.0-Flash",
        litellm_model="gemini/gemini-2.0-flash-exp",
        provider=ModelProvider.GOOGLE,
        knowledge_cutoff="2024-08-01",
        description="Google Gemini 2.0 Flash",
        price_per_1m_input=0.10,
        price_per_1m_output=0.40,
    ),
    "gemini-1.5-pro": ModelConfig(
        name="Gemini-1.5-Pro",
        litellm_model="gemini/gemini-1.5-pro",
        provider=ModelProvider.GOOGLE,
        knowledge_cutoff="2024-04-01",
        description="Google Gemini 1.5 Pro",
        price_per_1m_input=1.25,
        price_per_1m_output=5.00,
    ),
}


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    dataset_name: str = "2084Collective/prediction-markets-historical-v5-cleaned"
    knowledge_cutoff: str | None = None  # Auto-computed from models if None
    num_samples: int = 200
    output_dir: str = "kalshibench_results"
    seed: int = 42
    max_concurrent: int = 10  # Max concurrent API calls
    confidence_bins: int = 10  # For calibration analysis


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
    total_cost_usd: Optional[float] = None
    
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
        return asdict(self.config)
    
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
        
        try:
            response = await acompletion(
                model=self.config.litellm_model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            raw_output = response.choices[0].message.content
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            answer, confidence, reasoning = self._parse_response(raw_output)
            
            return ModelPrediction(
                question_id=question.id,
                predicted_answer=answer,
                predicted_probability=confidence,
                reasoning=reasoning,
                raw_output=raw_output,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return ModelPrediction(
                question_id=question.id,
                predicted_answer=None,
                predicted_probability=0.5,
                reasoning="",
                raw_output="",
                latency_ms=latency_ms,
                error=str(e),
            )


class KalshiDataLoader(BaseDataLoader):
    """Load prediction market data from HuggingFace dataset."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def load(self) -> list[MarketQuestion]:
        """Load and filter market questions."""
        logger.info(f"📥 Loading dataset: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name, split="train")
        logger.info(f"   Raw dataset size: {len(dataset):,}")
        
        # Filter by date (post-knowledge-cutoff)
        logger.info(f"🔍 Filtering by date >= {self.config.knowledge_cutoff}")
        def filter_by_date(example):
            close_time = example.get("close_time", "")
            return close_time and close_time >= self.config.knowledge_cutoff
        
        dataset = dataset.filter(filter_by_date, desc="Filtering by date")
        logger.info(f"   After date filter: {len(dataset):,}")
        
        # Deduplicate by series_ticker
        logger.info("🔄 Deduplicating by series_ticker")
        seen_tickers = set()
        def deduplicate(example):
            ticker = example.get("series_ticker", "")
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                return True
            return False
        
        dataset = dataset.filter(deduplicate, desc="Deduplicating")
        logger.info(f"   After deduplication: {len(dataset):,}")
        
        # Convert to MarketQuestion objects
        logger.info("📝 Converting to MarketQuestion objects")
        questions = []
        for item in tqdm(dataset, desc="Processing questions", unit="q"):
            winning_outcome = item.get("winning_outcome", "").lower()
            if winning_outcome not in ["yes", "no"]:
                continue
            
            questions.append(MarketQuestion(
                id=f"kalshi_{len(questions)}_{item.get('series_ticker', 'unknown')}",
                question=item.get("question", ""),
                description=item.get("description", "")[:2000],  # Truncate long descriptions
                category=item.get("category", "unknown"),
                close_time=item.get("close_time", ""),
                ground_truth=winning_outcome,
                market_probability=item.get("last_price"),  # If available
            ))
        
        logger.info(f"   Valid questions: {len(questions):,}")
        
        # Sample
        np.random.seed(self.config.seed)
        if len(questions) > self.config.num_samples:
            logger.info(f"🎲 Sampling {self.config.num_samples:,} questions from {len(questions):,}")
            indices = np.random.choice(len(questions), self.config.num_samples, replace=False)
            questions = [questions[i] for i in indices]
        
        # Show category distribution
        categories = defaultdict(int)
        for q in questions:
            categories[q.category] += 1
        logger.info(f"   Category distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"      {cat}: {count}")
        if len(categories) > 5:
            logger.info(f"      ... and {len(categories) - 5} more categories")
        
        logger.info(f"✅ Loaded {len(questions):,} questions")
        return questions


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
        self.start_time = None
    
    def load_data(self):
        """Load benchmark data."""
        self.questions = self.data_loader.load()
        return self
    
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
        
        # Build result
        result = EvaluationResult(
            model_name=model_config.name,
            model_config=model.get_config(),
            timestamp=datetime.now().isoformat(),
            num_samples=len(self.questions),
            predictions=[asdict(p) for p in predictions],
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
    
    async def run(self, model_keys: list[str]) -> dict[str, EvaluationResult]:
        """Run benchmark on specified models."""
        self.start_time = datetime.now()
        
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
        
        # Count valid models
        valid_models = [key for key in model_keys if key in MODELS]
        logger.info("")
        logger.info(f"🚀 Starting evaluation of {len(valid_models)} models")
        logger.info(f"   Total predictions to make: {len(valid_models) * len(self.questions):,}")
        
        for i, key in enumerate(model_keys, 1):
            if key not in MODELS:
                logger.warning(f"⚠️  Unknown model '{key}', skipping")
                continue
            
            logger.info(f"\n[{i}/{len(valid_models)}] Starting {MODELS[key].name}...")
            await self.evaluate_model(MODELS[key])
        
        # Final summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏁 BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   Total time: {elapsed/60:.1f} minutes")
        logger.info(f"   Models evaluated: {len(self.results)}")
        logger.info(f"   Questions per model: {len(self.questions):,}")
        
        return self.results
    
    def save_results(self):
        """Save all results to disk."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("💾 SAVING RESULTS")
        logger.info("=" * 60)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save individual model results
        logger.info("📄 Saving individual model results...")
        for name, result in tqdm(self.results.items(), desc="Saving models", unit="model"):
            path = os.path.join(self.config.output_dir, f"{name.lower().replace(' ', '_')}_{timestamp}.json")
            with open(path, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            logger.debug(f"   Saved: {path}")
        
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
        
        # 4. Generate paper prompts
        self._save_paper_prompts(timestamp)
        
        # 5. Generate markdown report
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
        
        # Methods section prompt
        methods_prompt = f"""You are a scientific writer preparing the Methods section for a NeurIPS Datasets & Benchmarks paper.

## Benchmark: KalshiBench

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
{json.dumps([{"name": k, "provider": v["provider"]} for k, v in [(name, asdict(MODELS[name.lower().replace('-', '_').replace('.', '_').replace(' ', '-')])) for name in self.results.keys() if name.lower().replace('-', '_').replace('.', '_').replace(' ', '-') in MODELS] if v], indent=2)}

Write a formal Methods section (3-4 paragraphs) suitable for NeurIPS covering:
1. Benchmark construction and rationale
2. Evaluation protocol and metrics
3. Model selection criteria
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

**Paragraph 2: Calibration Analysis (KEY CONTRIBUTION)**
- This is the main contribution of KalshiBench
- Compare Brier Scores across models (THE key metric for forecasting)
- Discuss ECE - which models are best calibrated?
- Analyze gap between accuracy and calibration (a model can be accurate but poorly calibrated)

**Paragraph 3: Overconfidence Analysis**
- Report overconfidence rates
- Which models are most/least overconfident?
- Discuss implications for deployment

**Paragraph 4: Category Breakdown**
- Are there categories where models struggle?
- Do different models have different strengths?

**Paragraph 5: Key Findings**
- Summarize main takeaways
- What does this reveal about current LLM capabilities?
- Implications for using LLMs for forecasting

Use precise numbers, scientific language, and draw meaningful conclusions. Create 1-2 tables summarizing key results.
"""
        
        # Save prompts
        methods_path = os.path.join(self.config.output_dir, f"paper_methods_prompt_{timestamp}.txt")
        results_path = os.path.join(self.config.output_dir, f"paper_results_prompt_{timestamp}.txt")
        
        with open(methods_path, "w") as f:
            f.write(methods_prompt)
        print(f"Saved: {methods_path}")
        
        with open(results_path, "w") as f:
            f.write(results_prompt)
        print(f"Saved: {results_path}")
    
    def _save_markdown_report(self, timestamp: str):
        """Save a human-readable markdown report."""
        summary = self._generate_summary()
        
        # Build tables
        accuracy_table = "| Model | Accuracy | Macro F1 | Brier Score | ECE | Parse Rate |\n"
        accuracy_table += "|-------|----------|----------|-------------|-----|------------|\n"
        for name, metrics in summary["models"].items():
            accuracy_table += f"| {name} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.3f} | {metrics['brier_score']:.4f} | {metrics['ece']:.4f} | {metrics['parse_rate']:.2%} |\n"
        
        calibration_table = "| Model | Brier Score | BSS | ECE | MCE | Log Loss | Overconf@80% |\n"
        calibration_table += "|-------|-------------|-----|-----|-----|----------|---------------|\n"
        for name, metrics in summary["models"].items():
            overconf = f"{metrics['overconfidence_rate_80']:.2%}" if metrics['overconfidence_rate_80'] else "N/A"
            calibration_table += f"| {name} | {metrics['brier_score']:.4f} | {metrics['brier_skill_score']:.4f} | {metrics['ece']:.4f} | {metrics['mce']:.4f} | {metrics['log_loss']:.4f} | {overconf} |\n"
        
        # Category breakdown for best model
        best_model = summary["leaderboard"]["by_accuracy"][0]["model"] if summary["leaderboard"]["by_accuracy"] else None
        category_section = ""
        if best_model and best_model in self.results:
            category_section = f"\n### Category Breakdown ({best_model})\n\n"
            category_section += "| Category | Accuracy | Brier Score | Count |\n"
            category_section += "|----------|----------|-------------|-------|\n"
            for cat, data in self.results[best_model].per_category.items():
                brier = f"{data['brier_score']:.4f}" if data['brier_score'] else "N/A"
                category_section += f"| {cat} | {data['accuracy']:.2%} | {brier} | {data['count']} |\n"
        
        report = f"""# KalshiBench Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Benchmark Version:** 1.0

## Overview

KalshiBench evaluates language model forecasting ability using temporally-filtered prediction market questions from Kalshi.

- **Total Questions:** {len(self.questions)}
- **Knowledge Cutoff:** {self.config.knowledge_cutoff}
- **Models Evaluated:** {len(self.results)}

## Main Results

### Classification Performance

{accuracy_table}

### Calibration Analysis

{calibration_table}

**Key Metrics:**
- **Brier Score**: Lower is better (0 = perfect, 1 = worst)
- **BSS (Brier Skill Score)**: Improvement over base rate (higher is better)
- **ECE**: Expected Calibration Error (lower is better)
- **Overconf@80%**: Rate of wrong predictions when confidence > 80%

## Leaderboards

### By Accuracy
{chr(10).join([f"{i+1}. **{m['model']}**: {m['accuracy']:.2%}" for i, m in enumerate(summary['leaderboard']['by_accuracy'][:5])])}

### By Calibration (Brier Score, lower is better)
{chr(10).join([f"{i+1}. **{m['model']}**: {m['brier_score']:.4f}" for i, m in enumerate(summary['leaderboard']['by_brier_score'][:5])])}

### By ECE (lower is better)
{chr(10).join([f"{i+1}. **{m['model']}**: {m['ece']:.4f}" for i, m in enumerate(summary['leaderboard']['by_calibration'][:5])])}
{category_section}
## Dataset Statistics

- **Date Range:** {min(q.close_time for q in self.questions)} to {max(q.close_time for q in self.questions)}
- **Ground Truth Distribution:**
  - Yes: {sum(1 for q in self.questions if q.ground_truth == "yes")} ({sum(1 for q in self.questions if q.ground_truth == "yes")/len(self.questions):.1%})
  - No: {sum(1 for q in self.questions if q.ground_truth == "no")} ({sum(1 for q in self.questions if q.ground_truth == "no")/len(self.questions):.1%})

## Files Generated

- `summary_*.json`: Aggregated results for all models
- `<model>_*.json`: Individual model results with predictions
- `metadata_*.json`: Benchmark configuration and statistics
- `paper_methods_prompt_*.txt`: Prompt for generating Methods section
- `paper_results_prompt_*.txt`: Prompt for generating Results section

## Citation

```bibtex
@misc{{kalshibench2024,
  title={{KalshiBench: Evaluating LLM Forecasting Calibration via Prediction Markets}},
  year={{2024}},
  note={{Evaluation of {len(self.results)} models on {len(self.questions)} prediction market questions}}
}}
```
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
Available models:
  OpenAI:    gpt-5.2, gpt-5.1, gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini
  Anthropic: claude-opus-4.5, claude-sonnet-4.5, claude-3-5-sonnet, claude-3-5-haiku
  Google:    gemini-2.0-flash, gemini-1.5-pro
  Together:  qwen-2.5-72b, qwen-2.5-7b, qwen-qwq-32b,
             llama-3.3-70b, llama-3.1-405b, llama-3.1-8b,
             deepseek-v3, deepseek-r1, mistral-large

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
    )
    
    print("\n" + "=" * 60)
    print("KalshiBench - LLM Forecasting Calibration Benchmark")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Samples: {args.samples}")
    print(f"Knowledge Cutoff: {args.cutoff}")
    print(f"Output: {args.output}")
    
    # Run benchmark
    runner = KalshiBenchRunner(config)
    asyncio.run(runner.run(args.models))
    
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

