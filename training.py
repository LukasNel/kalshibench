
def main(output_dir: str):
    # Patch GRPO for Unsloth
    import os
    import re
    import json
    import torch
    import wandb
    import numpy as np
    from datetime import datetime
    from collections import defaultdict
    from datasets import load_dataset
    from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
    PatchFastRL("GRPO", FastLanguageModel)

    # ==================================================
    # Configuration
    # ==================================================
    class Config:
        # Model settings
        MODEL_NAME = "unsloth/Qwen3-4B"
        MAX_SEQ_LENGTH = 4096
        LORA_RANK = 32
        LOAD_IN_4BIT = True
        
        # Training settings
        LEARNING_RATE = 5e-6
        MAX_STEPS = 200
        SAVE_STEPS = 10
        NUM_GENERATIONS = 6
        MAX_PROMPT_LENGTH = 512
        GRADIENT_ACCUMULATION = 4
        
        # Dataset settings
        DATASET_NAME = "2084Collective/prediction-markets-historical-v5-cleaned"
        QWEN3_RELEASE_DATE = "2025-04-29"
        
        # Evaluation settings
        EVAL_SAMPLES = 100
        ROLLING_EVAL_INTERVAL = 25  # Evaluate every N steps
        BOOTSTRAP_SAMPLES = 1000  # For confidence intervals
        
        # WandB settings
        WANDB_PROJECT = "prediction-market-grpo"
        WANDB_RUN_NAME = f"qwen3-4b-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Output
        OUTPUT_DIR = os.path.join(output_dir, "outputs")
        CHECKPOINT_DIR = os.path.join(output_dir, "checkpoints")
        RESULTS_DIR = os.path.join(output_dir, "results")

    config = Config()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ==================================================
    # Initialize WandB
    # ==================================================
    wandb_run = wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "model_name": config.MODEL_NAME,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "lora_rank": config.LORA_RANK,
            "learning_rate": config.LEARNING_RATE,
            "max_steps": config.MAX_STEPS,
            "num_generations": config.NUM_GENERATIONS,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION,
        }
    )

    # ==================================================
    # Comprehensive Results Tracker for Paper
    # ==================================================
    class PaperResultsTracker:
        """Track all metrics needed for a comprehensive NeurIPS paper."""
        
        def __init__(self, config):
            self.config = config
            self.training_history = []
            self.rolling_evaluations = []
            self.final_evaluation = None
            self.baseline_evaluation = None
            self.per_category_results = defaultdict(list)
            self.calibration_data = {}  # Will store baseline vs final calibration
            self.benchmark_results = {}  # Will store post-training benchmark results
            self.baseline_benchmark_results = {}  # Will store pre-training benchmark results
            self.benchmark_comparison = {}  # Will store before/after comparison
            self.start_time = datetime.now()
            
        def log_training_step(self, step, metrics):
            """Log training metrics at each step."""
            self.training_history.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **metrics
            })
            
        def log_rolling_evaluation(self, step, metrics):
            """Log periodic evaluation during training."""
            self.rolling_evaluations.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **metrics
            })
            
        def compute_classification_metrics(self, predictions, ground_truth):
            """Compute precision, recall, F1 for binary classification."""
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == "yes" and g == "yes")
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "yes" and g == "no")
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p == "no" and g == "yes")
            tn = sum(1 for p, g in zip(predictions, ground_truth) if p == "no" and g == "no")
            
            precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes) if (precision_yes + recall_yes) > 0 else 0
            
            precision_no = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall_no = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_no = 2 * precision_no * recall_no / (precision_no + recall_no) if (precision_no + recall_no) > 0 else 0
            
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
            macro_f1 = (f1_yes + f1_no) / 2
            
            return {
                "accuracy": accuracy,
                "precision_yes": precision_yes,
                "recall_yes": recall_yes,
                "f1_yes": f1_yes,
                "precision_no": precision_no,
                "recall_no": recall_no,
                "f1_no": f1_no,
                "macro_f1": macro_f1,
                "confusion_matrix": {
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn
                }
            }
        
        def compute_brier_score(self, predicted_probs, ground_truth):
            """
            Compute Brier Score - key metric for prediction market forecasting.
            Lower is better. Perfect score = 0, worst = 1.
            """
            scores = []
            for prob, actual in zip(predicted_probs, ground_truth):
                actual_binary = 1.0 if actual == "yes" else 0.0
                scores.append((prob - actual_binary) ** 2)
            return np.mean(scores) if scores else None
        
        def compute_calibration_metrics(self, predicted_probs, ground_truth, n_bins=10):
            """
            Compute Expected Calibration Error (ECE) and reliability diagram data.
            Critical for evaluating prediction market forecasters.
            """
            bins = np.linspace(0, 1, n_bins + 1)
            bin_data = []
            
            for i in range(n_bins):
                bin_lower, bin_upper = bins[i], bins[i + 1]
                bin_mask = [(p >= bin_lower and p < bin_upper) for p in predicted_probs]
                
                if sum(bin_mask) > 0:
                    bin_probs = [p for p, m in zip(predicted_probs, bin_mask) if m]
                    bin_actuals = [1.0 if g == "yes" else 0.0 for g, m in zip(ground_truth, bin_mask) if m]
                    
                    avg_confidence = np.mean(bin_probs)
                    avg_accuracy = np.mean(bin_actuals)
                    bin_count = len(bin_probs)
                    
                    bin_data.append({
                        "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                        "avg_confidence": avg_confidence,
                        "avg_accuracy": avg_accuracy,
                        "count": bin_count
                    })
            
            # Compute ECE
            total_samples = len(predicted_probs)
            ece = sum(
                (b["count"] / total_samples) * abs(b["avg_confidence"] - b["avg_accuracy"])
                for b in bin_data
            ) if total_samples > 0 else None
            
            return {
                "ece": ece,
                "reliability_diagram_data": bin_data,
                "n_bins": n_bins
            }
        
        def compute_advanced_calibration_metrics(self, predicted_probs, ground_truth):
            """
            Compute comprehensive calibration metrics for Research Angle 1:
            "Can RL teach LLMs epistemic calibration?"
            """
            actuals = [1.0 if g == "yes" else 0.0 for g in ground_truth]
            probs = np.array(predicted_probs)
            actuals = np.array(actuals)
            
            metrics = {}
            
            # 1. Brier Score (already have, but include here for completeness)
            metrics["brier_score"] = np.mean((probs - actuals) ** 2)
            
            # 2. Brier Skill Score (relative to climatological baseline)
            # Climatology = always predict base rate
            base_rate = np.mean(actuals)
            brier_climatology = np.mean((base_rate - actuals) ** 2)
            if brier_climatology > 0:
                metrics["brier_skill_score"] = 1 - (metrics["brier_score"] / brier_climatology)
            else:
                metrics["brier_skill_score"] = 0
            
            # 3. Log Loss (Cross-Entropy) - more sensitive to confident wrong predictions
            eps = 1e-15
            probs_clipped = np.clip(probs, eps, 1 - eps)
            log_loss = -np.mean(actuals * np.log(probs_clipped) + (1 - actuals) * np.log(1 - probs_clipped))
            metrics["log_loss"] = log_loss
            
            # 4. Overconfidence metrics
            # Average confidence when wrong
            wrong_mask = (probs > 0.5) != (actuals > 0.5)
            if np.sum(wrong_mask) > 0:
                metrics["avg_confidence_when_wrong"] = np.mean(np.abs(probs[wrong_mask] - 0.5)) + 0.5
            else:
                metrics["avg_confidence_when_wrong"] = None
            
            # Average confidence when right
            right_mask = ~wrong_mask
            if np.sum(right_mask) > 0:
                metrics["avg_confidence_when_right"] = np.mean(np.abs(probs[right_mask] - 0.5)) + 0.5
            else:
                metrics["avg_confidence_when_right"] = None
            
            # 5. Overconfidence ratio: how often model is >X% confident but wrong
            for threshold in [0.7, 0.8, 0.9]:
                high_conf_mask = (probs > threshold) | (probs < (1 - threshold))
                if np.sum(high_conf_mask) > 0:
                    high_conf_wrong = np.sum(high_conf_mask & wrong_mask)
                    metrics[f"overconfidence_rate_{int(threshold*100)}"] = high_conf_wrong / np.sum(high_conf_mask)
                else:
                    metrics[f"overconfidence_rate_{int(threshold*100)}"] = None
            
            # 6. Maximum Calibration Error (MCE) - worst bin
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            max_ce = 0
            for i in range(n_bins):
                bin_mask = (probs >= bins[i]) & (probs < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    bin_conf = np.mean(probs[bin_mask])
                    bin_acc = np.mean(actuals[bin_mask])
                    max_ce = max(max_ce, abs(bin_conf - bin_acc))
            metrics["mce"] = max_ce
            
            # 7. Adaptive Calibration Error (ACE) - equal mass bins
            sorted_indices = np.argsort(probs)
            n_per_bin = len(probs) // n_bins
            ace = 0
            for i in range(n_bins):
                start_idx = i * n_per_bin
                end_idx = start_idx + n_per_bin if i < n_bins - 1 else len(probs)
                bin_indices = sorted_indices[start_idx:end_idx]
                if len(bin_indices) > 0:
                    bin_conf = np.mean(probs[bin_indices])
                    bin_acc = np.mean(actuals[bin_indices])
                    ace += len(bin_indices) * abs(bin_conf - bin_acc)
            metrics["ace"] = ace / len(probs) if len(probs) > 0 else None
            
            # 8. Calibration curve area (deviation from perfect calibration line)
            # Perfect calibration = diagonal line, measure area between model curve and diagonal
            reliability_data = []
            for i in range(n_bins):
                bin_mask = (probs >= bins[i]) & (probs < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    reliability_data.append({
                        "confidence": np.mean(probs[bin_mask]),
                        "accuracy": np.mean(actuals[bin_mask]),
                        "count": np.sum(bin_mask)
                    })
            
            if reliability_data:
                calibration_area = np.mean([abs(d["confidence"] - d["accuracy"]) for d in reliability_data])
                metrics["calibration_curve_area"] = calibration_area
            else:
                metrics["calibration_curve_area"] = None
            
            # 9. Resolution (ability to distinguish outcomes)
            # Higher is better - measures how much predictions deviate from base rate
            resolution = np.mean((probs - base_rate) ** 2)
            metrics["resolution"] = resolution
            
            # 10. Reliability (calibration component of Brier score)
            reliability = 0
            for i in range(n_bins):
                bin_mask = (probs >= bins[i]) & (probs < bins[i + 1])
                if np.sum(bin_mask) > 0:
                    bin_conf = np.mean(probs[bin_mask])
                    bin_acc = np.mean(actuals[bin_mask])
                    reliability += np.sum(bin_mask) * (bin_conf - bin_acc) ** 2
            metrics["reliability"] = reliability / len(probs) if len(probs) > 0 else None
            
            return metrics
        
        def compute_bootstrap_ci(self, values, confidence=0.95, n_bootstrap=1000):
            """Compute bootstrap confidence interval for statistical significance."""
            if len(values) == 0:
                return None, None, None
            
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            alpha = 1 - confidence
            lower = np.percentile(bootstrap_means, alpha / 2 * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
            mean = np.mean(values)
            
            return mean, lower, upper
        
        def compute_mcnemar_test(self, baseline_correct, model_correct):
            """
            McNemar's test for comparing two classifiers on same data.
            Returns p-value for statistical significance.
            """
            # Count discordant pairs
            b = sum(1 for bc, mc in zip(baseline_correct, model_correct) if not bc and mc)  # baseline wrong, model right
            c = sum(1 for bc, mc in zip(baseline_correct, model_correct) if bc and not mc)  # baseline right, model wrong
            
            # McNemar's test statistic (with continuity correction)
            if b + c == 0:
                return 1.0  # No difference
            
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            
            # Approximate p-value from chi-squared distribution with 1 df
            from scipy import stats
            try:
                p_value = 1 - stats.chi2.cdf(chi2, df=1)
            except:
                # Fallback if scipy not available
                p_value = None
            
            return p_value
        
        def generate_paper_output(self):
            """Generate comprehensive output for paper methods/results sections."""
            output = {
                "metadata": {
                    "experiment_date": self.start_time.isoformat(),
                    "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                    "model_name": self.config.MODEL_NAME,
                    "dataset": self.config.DATASET_NAME,
                    "wandb_run_name": self.config.WANDB_RUN_NAME,
                },
                "hyperparameters": {
                    "learning_rate": self.config.LEARNING_RATE,
                    "max_steps": self.config.MAX_STEPS,
                    "lora_rank": self.config.LORA_RANK,
                    "num_generations": self.config.NUM_GENERATIONS,
                    "gradient_accumulation": self.config.GRADIENT_ACCUMULATION,
                    "max_seq_length": self.config.MAX_SEQ_LENGTH,
                },
                "training_curve": self.training_history,
                "rolling_evaluations": self.rolling_evaluations,
                "baseline_evaluation": self.baseline_evaluation,
                "final_evaluation": self.final_evaluation,
                "per_category_results": dict(self.per_category_results),
                "calibration_analysis": self.calibration_data,
                "benchmark_results": getattr(self, 'benchmark_results', {}),
                "baseline_benchmark_results": getattr(self, 'baseline_benchmark_results', {}),
                "benchmark_comparison": getattr(self, 'benchmark_comparison', {}),
            }
            return output
        
        def generate_methods_section_prompt(self):
            """Generate a prompt for LLM to create methods section."""
            output = self.generate_paper_output()
            
            prompt = f"""You are a scientific writer preparing a methods section for a NeurIPS paper on using Group Relative Policy Optimization (GRPO) for prediction market forecasting.

Based on the following experimental configuration and results, write a comprehensive Methods section:

## Experimental Configuration

**Model Architecture:**
- Base Model: {output['metadata']['model_name']}
- Fine-tuning Method: LoRA (Low-Rank Adaptation) with rank {output['hyperparameters']['lora_rank']}
- Training Algorithm: GRPO (Group Relative Policy Optimization)

**Training Configuration:**
- Learning Rate: {output['hyperparameters']['learning_rate']}
- Training Steps: {output['hyperparameters']['max_steps']}
- Number of Generations per Step: {output['hyperparameters']['num_generations']}
- Gradient Accumulation Steps: {output['hyperparameters']['gradient_accumulation']}
- Maximum Sequence Length: {output['hyperparameters']['max_seq_length']}

**Dataset:**
- Source: {output['metadata']['dataset']}
- Filtering: Markets closing after Qwen3 release date (2025-04-29) to avoid training data contamination

**Reward Functions:**
1. Correctness Reward: +2.0 for correct prediction, -1.0 for incorrect, -0.5 for unparseable
2. Format Reward: +1.0 for proper <think>...</think><answer>...</answer> format
3. Reasoning Length Reward: Encourages detailed reasoning (50-300+ words)

Write a formal methods section (2-3 paragraphs) suitable for NeurIPS that covers:
1. Model architecture and fine-tuning approach
2. Training procedure and reward design
3. Dataset preparation and filtering methodology
"""
            return prompt
        
        def generate_results_section_prompt(self):
            """Generate a prompt for LLM to create results section covering both research angles."""
            output = self.generate_paper_output()
            
            # Extract key metrics
            final_metrics = output.get('final_evaluation', {})
            baseline_metrics = output.get('baseline_evaluation', {})
            calibration = output.get('calibration_analysis', {})
            benchmarks = output.get('benchmark_results', {})
            
            # Clean metrics for readability
            def clean_metrics(m):
                if not m:
                    return {}
                return {k: v for k, v in m.items() if k not in ['raw_data', 'calibration_data']}
            
            prompt = f"""You are a scientific writer preparing a results section for a NeurIPS paper with TWO main research contributions:

1. **Research Angle 1**: Can RL (GRPO) teach LLMs epistemic calibration?
2. **Research Angle 2**: Does forecasting training transfer to general reasoning?

Based on the experimental results below, write a comprehensive Results section (4-5 paragraphs):

## PRIMARY TASK: Prediction Market Forecasting

**Baseline (Pre-GRPO) Performance:**
{json.dumps(clean_metrics(baseline_metrics), indent=2, default=str)}

**Final (Post-GRPO) Performance:**
{json.dumps(clean_metrics(final_metrics), indent=2, default=str)}

**Training Progression:**
{json.dumps(output.get('rolling_evaluations', []), indent=2, default=str)}

## RESEARCH ANGLE 1: Calibration Analysis

**Advanced Calibration Metrics (Baseline vs Final):**
{json.dumps(calibration, indent=2, default=str)}

Key metrics to discuss:
- Brier Score (lower is better) - gold standard for probabilistic forecasting
- ECE/MCE/ACE - Expected/Maximum/Adaptive Calibration Error  
- Overconfidence rates at 70/80/90% thresholds
- Reliability vs Resolution decomposition
- Log Loss (penalizes confident wrong predictions more)

## RESEARCH ANGLE 2: Standard Benchmarks (Transfer Learning)

**Benchmark Results (Post-GRPO model):**
{json.dumps(benchmarks, indent=2, default=str)}

Standard benchmarks used (subsets of 50 examples each):
- **TruthfulQA**: Tests truthfulness and resistance to common misconceptions
- **ARC-Easy**: Science reasoning from AI2 Reasoning Challenge
- **MMLU**: General knowledge (miscellaneous subset)
- **HellaSwag**: Commonsense reasoning / sentence completion
- **WinoGrande**: Commonsense coreference resolution

## WRITING INSTRUCTIONS

Structure your Results section as:

**Paragraph 1: Main Forecasting Results**
- Report accuracy/F1 improvement with confidence intervals
- Statistical significance (McNemar's test p-value)
- Highlight that this is the primary task

**Paragraph 2: Calibration Improvements (Research Angle 1)**
- Focus on Brier Score improvement (THE key metric for forecasting)
- Discuss ECE reduction - model becomes better calibrated
- Analyze overconfidence reduction
- This supports "RL can teach epistemic calibration"

**Paragraph 3: Training Dynamics**
- How accuracy/calibration evolved during training
- Any interesting patterns in the learning curve
- When did improvements plateau?

**Paragraph 4: Transfer Learning (Research Angle 2)**
- Report accuracies on standard benchmarks: TruthfulQA, ARC, MMLU, HellaSwag, WinoGrande
- Compare to known baseline accuracies for the base model (look up Qwen3-4B benchmarks)
- TruthfulQA is particularly relevant as it tests epistemic virtues
- Be cautious about strong claims; ideally need pre-GRPO benchmark comparison

**Paragraph 5: Discussion/Limitations**
- Interpret findings for both research angles
- Note limitations (scale, baseline comparisons needed)
- Suggest future directions

Use precise numbers, scientific language, and confidence intervals. Frame improvements as evidence for the two research hypotheses.
"""
            return prompt

    results_tracker = PaperResultsTracker(config)

    # ==================================================
    # Load Model
    # ==================================================
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT,
        fast_inference=False,  # Use HuggingFace generate, not vLLM
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("Model loaded successfully!")

    # ==================================================
    # Load and Process Dataset
    # ==================================================
    print("Loading dataset...")
    dataset = load_dataset(config.DATASET_NAME, split="train")
    print(f"Original dataset size: {len(dataset)}")

    def filter_by_date(example):
        try:
            close_time = example.get("close_time", "")
            if close_time and close_time >= config.QWEN3_RELEASE_DATE:
                return True
        except:
            pass
        return False
    
    dataset = dataset.filter(filter_by_date)
    print(f"After date filter: {len(dataset)}")

    # Deduplicate by series_ticker
    seen_tickers = set()

    def deduplicate_by_series_ticker(example):
        ticker = example.get("series_ticker", "")
        if ticker and ticker not in seen_tickers:
            seen_tickers.add(ticker)
            return True
        return False

    dataset = dataset.filter(deduplicate_by_series_ticker)
    print(f"After deduplication: {len(dataset)}")

    # ==================================================
    # System Prompt and Formatting
    # ==================================================
    SYSTEM_PROMPT = """You are an expert prediction market analyst. Given a prediction market question and its description, predict whether the outcome will be "yes" or "no".

Think step by step about the factors that influence this prediction, then provide your final answer.

Respond in the following format:
<think>
[Your reasoning about the prediction market question]
</think>
<answer>
[yes or no]
</answer>"""

    def format_prompt(question, description):
        """Format the prediction market question as a prompt."""
        prompt = f"""Question: {question}

Description: {description}

Analyze this prediction market and predict the outcome."""
        return prompt

    def extract_answer(text):
        """Extract the answer from model output."""
        match = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        text_lower = text.lower()
        if 'the answer is yes' in text_lower or 'outcome: yes' in text_lower:
            return 'yes'
        if 'the answer is no' in text_lower or 'outcome: no' in text_lower:
            return 'no'
        
        yes_count = text_lower.count('yes')
        no_count = text_lower.count('no')
        if yes_count > no_count:
            return 'yes'
        elif no_count > yes_count:
            return 'no'
        
        return None

    def get_token_probability_confidence(model, tokenizer, prompt):
        """
        Get confidence by comparing token probabilities for 'yes' vs 'no'.
        Returns P(yes) based on the model's logits.
        """
        FastLanguageModel.for_inference(model)
        
        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_PROMPT_LENGTH
        ).to(model.device)
        
        # Get logits for the next token
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]  # Last position
        
        # Get token IDs for "yes" and "no" (try multiple variants)
        yes_tokens = []
        no_tokens = []
        for word in ["yes", "Yes", "YES", " yes", " Yes"]:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if tokens:
                yes_tokens.append(tokens[0])
        for word in ["no", "No", "NO", " no", " No"]:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if tokens:
                no_tokens.append(tokens[0])
        
        # Remove duplicates
        yes_tokens = list(set(yes_tokens))
        no_tokens = list(set(no_tokens))
        
        if not yes_tokens or not no_tokens:
            return 0.5  # Fallback
        
        # Get max logit for yes and no token variants
        yes_logit = max(next_token_logits[t].item() for t in yes_tokens)
        no_logit = max(next_token_logits[t].item() for t in no_tokens)
        
        # Convert to probability using softmax over just yes/no
        logits = torch.tensor([yes_logit, no_logit])
        probs = torch.softmax(logits, dim=0)
        
        return probs[0].item()  # P(yes)

    def get_answer_with_confidence(model, tokenizer, prompt, generated_text):
        """
        Extract answer and get confidence from token probabilities.
        This is more principled than parsing text for confidence phrases.
        
        We construct a prompt that ends right before the answer token and
        measure P(yes) vs P(no).
        """
        # First, extract the answer from generated text
        answer = extract_answer(generated_text)
        
        # Find where the <answer> tag is and construct prompt up to that point
        # to measure token probabilities
        answer_prompt = prompt
        if "<answer>" in generated_text:
            # Add everything up to and including <answer>\n
            idx = generated_text.find("<answer>")
            prefix = generated_text[:idx + len("<answer>")]
            answer_prompt = prompt + prefix + "\n"
        
        # Get probability from token logits
        prob_yes = get_token_probability_confidence(model, tokenizer, answer_prompt)
        
        return answer, prob_yes

    # ==================================================
    # Prepare Dataset for GRPO
    # ==================================================
    def prepare_grpo_example(example):
        """Convert example to GRPO format."""
        question = example.get("question", "")
        description = example.get("description", "")
        winning_outcome = example.get("winning_outcome", "").lower()
        category = example.get("category", "unknown")
        
        if not question or winning_outcome not in ["yes", "no"]:
            return None
        
        prompt = format_prompt(question, description[:1000])
        
        return {
            "prompt": prompt,
            "answer": winning_outcome,
            "question": question,
            "category": category,
        }

    processed_dataset = dataset.map(prepare_grpo_example, remove_columns=dataset.column_names)
    processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
    processed_data = list(processed_dataset)
    print(f"Processed dataset size: {len(processed_data)}")

    # Split into train and eval
    np.random.seed(42)
    indices = np.random.permutation(len(processed_data))
    eval_size = min(config.EVAL_SAMPLES, len(processed_data) // 5)
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]
    
    eval_data_raw = [processed_data[i] for i in eval_indices]
    train_data_raw = [processed_data[i] for i in train_indices]
    
    print(f"Training set: {len(train_data_raw)}, Evaluation set: {len(eval_data_raw)}")

    from datasets import Dataset as HFDataset

    def create_grpo_dataset(data):
        """Create dataset in GRPO-compatible format."""
        formatted = []
        for item in data:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]
            formatted.append({
                "prompt": tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                ),
                "answer": item["answer"],
                "category": item.get("category", "unknown"),
            })
        return HFDataset.from_list(formatted)

    grpo_dataset = create_grpo_dataset(train_data_raw)
    eval_dataset = create_grpo_dataset(eval_data_raw)
    print(f"GRPO dataset ready: {len(grpo_dataset)} examples")

    # ==================================================
    # Reward Functions
    # ==================================================
    def correctness_reward_func(prompts, completions, answer, **kwargs):
        """Reward for predicting the correct outcome."""
        rewards = []
        for completion, correct_answer in zip(completions, answer):
            predicted = extract_answer(completion)
            if predicted == correct_answer:
                rewards.append(2.0)
            elif predicted is None:
                rewards.append(-0.5)
            else:
                rewards.append(-1.0)
        return rewards

    def format_reward_func(prompts, completions, **kwargs):
        """Reward for using the correct XML format."""
        rewards = []
        for completion in completions:
            has_think = '<think>' in completion and '</think>' in completion
            has_answer = '<answer>' in completion and '</answer>' in completion
            
            if has_think and has_answer:
                rewards.append(1.0)
            elif has_answer:
                rewards.append(0.3)
            else:
                rewards.append(-0.5)
        return rewards

    def reasoning_length_reward_func(prompts, completions, **kwargs):
        """Reward for appropriate reasoning length."""
        rewards = []
        for completion in completions:
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            if think_match:
                thinking = think_match.group(1)
                word_count = len(thinking.split())
                
                if word_count > 500:
                    rewards.append(0.2)
                elif word_count > 300:
                    rewards.append(0.1)
                else:
                    rewards.append(-0.2)
            else:
                rewards.append(-0.3)
        return rewards

    # ==================================================
    # Evaluation Functions (Fixed - using HuggingFace generate)
    # ==================================================
    def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
        """Generate response using HuggingFace generate (not vLLM fast_generate)."""
        FastLanguageModel.for_inference(model)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_PROMPT_LENGTH
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def evaluate_model(model, tokenizer, eval_data, num_samples=None, description="Evaluation"):
        """
        Comprehensive evaluation with all metrics needed for paper.
        """
        print(f"\n--- {description} ---")
        FastLanguageModel.for_inference(model)
        
        if num_samples is None:
            num_samples = len(eval_data)
        eval_subset = list(eval_data)[:num_samples]
        
        predictions = []
        ground_truths = []
        confidences = []
        categories = []
        correctness = []
        raw_outputs = []
        
        for i, item in enumerate(eval_subset):
            prompt = item["prompt"]
            correct_answer = item["answer"]
            category = item.get("category", "unknown")
            
            try:
                output = generate_response(model, tokenizer, prompt)
                # Use token probabilities for confidence (more principled)
                predicted, confidence = get_answer_with_confidence(model, tokenizer, prompt, output)
                
                predictions.append(predicted if predicted else "none")
                ground_truths.append(correct_answer)
                confidences.append(confidence)
                categories.append(category)
                correctness.append(predicted == correct_answer)
                raw_outputs.append(output[:500])  # Truncate for storage
                
                if (i + 1) % 10 == 0:
                    print(f"  Evaluated {i + 1}/{len(eval_subset)} samples...")
                    
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                predictions.append("error")
                ground_truths.append(correct_answer)
                confidences.append(0.5)
                categories.append(category)
                correctness.append(False)
                raw_outputs.append(str(e))
        
        # Filter out errors for metrics
        valid_mask = [p != "error" and p != "none" for p in predictions]
        valid_predictions = [p for p, v in zip(predictions, valid_mask) if v]
        valid_ground_truths = [g for g, v in zip(ground_truths, valid_mask) if v]
        valid_confidences = [c for c, v in zip(confidences, valid_mask) if v]
        valid_correctness = [c for c, v in zip(correctness, valid_mask) if v]
        
        # Compute all metrics
        metrics = results_tracker.compute_classification_metrics(valid_predictions, valid_ground_truths)
        
        # Brier Score
        brier_score = results_tracker.compute_brier_score(valid_confidences, valid_ground_truths)
        metrics["brier_score"] = brier_score
        
        # Calibration
        calibration = results_tracker.compute_calibration_metrics(valid_confidences, valid_ground_truths)
        metrics["ece"] = calibration["ece"]
        metrics["calibration_data"] = calibration
        
        # Bootstrap confidence intervals for accuracy
        if valid_correctness:
            acc_mean, acc_lower, acc_upper = results_tracker.compute_bootstrap_ci(
                [1.0 if c else 0.0 for c in valid_correctness],
                n_bootstrap=config.BOOTSTRAP_SAMPLES
            )
            metrics["accuracy_ci_lower"] = acc_lower
            metrics["accuracy_ci_upper"] = acc_upper
        
        # Parse rate
        metrics["parse_rate"] = sum(valid_mask) / len(predictions) if predictions else 0
        metrics["total_samples"] = len(eval_subset)
        metrics["valid_samples"] = len(valid_predictions)
        
        # Per-category breakdown
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for pred, gt, cat in zip(predictions, ground_truths, categories):
            if pred not in ["error", "none"]:
                category_metrics[cat]["total"] += 1
                if pred == gt:
                    category_metrics[cat]["correct"] += 1
        
        metrics["per_category"] = {
            cat: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "count": data["total"]
            }
            for cat, data in category_metrics.items()
        }
        
        # Store raw data for further analysis
        metrics["raw_data"] = {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "confidences": confidences,
            "correctness": correctness,
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.2%} (95% CI: [{metrics.get('accuracy_ci_lower', 0):.2%}, {metrics.get('accuracy_ci_upper', 0):.2%}])")
        print(f"  Macro F1: {metrics['macro_f1']:.3f}")
        print(f"  Brier Score: {metrics.get('brier_score', 'N/A'):.4f}" if metrics.get('brier_score') else "  Brier Score: N/A")
        print(f"  ECE: {metrics.get('ece', 'N/A'):.4f}" if metrics.get('ece') else "  ECE: N/A")
        print(f"  Parse Rate: {metrics['parse_rate']:.2%}")
        
        return metrics, valid_correctness

    def calculate_perplexity(model, tokenizer, texts, max_length=512):
        """Calculate perplexity on a set of texts."""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_length
                ).to(model.device)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    # ==================================================
    # Standard Benchmark Evaluation Function (Reusable)
    # ==================================================
    BENCHMARK_SAMPLE_SIZE = 50  # Samples per benchmark
    
    def evaluate_multiple_choice(model, tokenizer, prompt, choices):
        """
        Evaluate multiple choice by checking which choice token has highest probability.
        """
        FastLanguageModel.for_inference(model)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=config.MAX_PROMPT_LENGTH).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
        
        choice_probs = {}
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            letter_tokens = tokenizer.encode(letter, add_special_tokens=False)
            if letter_tokens:
                choice_probs[i] = next_token_logits[letter_tokens[0]].item()
        
        if choice_probs:
            return max(choice_probs, key=choice_probs.get)
        return 0
    
    def run_standard_benchmarks(model, tokenizer, description="Benchmarks"):
        """
        Run standard benchmarks: TruthfulQA, ARC-Easy, MMLU, HellaSwag, WinoGrande.
        Returns dict of benchmark results.
        """
        print(f"\n--- Running {description} ---")
        print("Benchmarks: TruthfulQA, ARC-Easy, MMLU, HellaSwag, WinoGrande")
        
        results = {}
        
        # 1. TruthfulQA
        print(f"\n  [{description}] TruthfulQA...")
        try:
            truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation")
            truthfulqa_subset = list(truthfulqa.shuffle(seed=42).select(range(min(BENCHMARK_SAMPLE_SIZE, len(truthfulqa)))))
            
            correct = 0
            for item in truthfulqa_subset:
                question = item["question"]
                choices = item["mc1_targets"]["choices"]
                labels = item["mc1_targets"]["labels"]
                correct_idx = labels.index(1)
                
                prompt = f"Question: {question}\n\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "\nAnswer with just the letter (A, B, C, etc.):"
                
                if evaluate_multiple_choice(model, tokenizer, prompt, choices) == correct_idx:
                    correct += 1
            
            accuracy = correct / len(truthfulqa_subset)
            results["TruthfulQA"] = {"accuracy": accuracy, "correct": correct, "total": len(truthfulqa_subset)}
            print(f"    TruthfulQA: {accuracy:.2%}")
        except Exception as e:
            print(f"    TruthfulQA failed: {e}")
            results["TruthfulQA"] = {"error": str(e)}
        
        # 2. ARC-Easy
        print(f"  [{description}] ARC-Easy...")
        try:
            arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            arc_subset = list(arc.shuffle(seed=42).select(range(min(BENCHMARK_SAMPLE_SIZE, len(arc)))))
            
            correct = 0
            for item in arc_subset:
                question = item["question"]
                choices = item["choices"]["text"]
                choice_labels = item["choices"]["label"]
                correct_idx = choice_labels.index(item["answerKey"])
                
                prompt = f"Question: {question}\n\n"
                for i, (label, choice) in enumerate(zip(choice_labels, choices)):
                    prompt += f"{label}. {choice}\n"
                prompt += "\nAnswer with just the letter:"
                
                if evaluate_multiple_choice(model, tokenizer, prompt, choices) == correct_idx:
                    correct += 1
            
            accuracy = correct / len(arc_subset)
            results["ARC-Easy"] = {"accuracy": accuracy, "correct": correct, "total": len(arc_subset)}
            print(f"    ARC-Easy: {accuracy:.2%}")
        except Exception as e:
            print(f"    ARC-Easy failed: {e}")
            results["ARC-Easy"] = {"error": str(e)}
        
        # 3. MMLU
        print(f"  [{description}] MMLU...")
        try:
            mmlu = load_dataset("cais/mmlu", "miscellaneous", split="test")
            mmlu_subset = list(mmlu.shuffle(seed=42).select(range(min(BENCHMARK_SAMPLE_SIZE, len(mmlu)))))
            
            correct = 0
            for item in mmlu_subset:
                question = item["question"]
                choices = item["choices"]
                correct_idx = item["answer"]
                
                prompt = f"Question: {question}\n\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "\nAnswer with just the letter (A, B, C, or D):"
                
                if evaluate_multiple_choice(model, tokenizer, prompt, choices) == correct_idx:
                    correct += 1
            
            accuracy = correct / len(mmlu_subset)
            results["MMLU"] = {"accuracy": accuracy, "correct": correct, "total": len(mmlu_subset)}
            print(f"    MMLU: {accuracy:.2%}")
        except Exception as e:
            print(f"    MMLU failed: {e}")
            results["MMLU"] = {"error": str(e)}
        
        # 4. HellaSwag
        print(f"  [{description}] HellaSwag...")
        try:
            hellaswag = load_dataset("Rowan/hellaswag", split="validation")
            hellaswag_subset = list(hellaswag.shuffle(seed=42).select(range(min(BENCHMARK_SAMPLE_SIZE, len(hellaswag)))))
            
            correct = 0
            for item in hellaswag_subset:
                context = item["ctx"]
                choices = item["endings"]
                correct_idx = int(item["label"])
                
                prompt = f"Complete the following:\n\n{context}\n\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "\nWhich ending makes the most sense? Answer with just the letter:"
                
                if evaluate_multiple_choice(model, tokenizer, prompt, choices) == correct_idx:
                    correct += 1
            
            accuracy = correct / len(hellaswag_subset)
            results["HellaSwag"] = {"accuracy": accuracy, "correct": correct, "total": len(hellaswag_subset)}
            print(f"    HellaSwag: {accuracy:.2%}")
        except Exception as e:
            print(f"    HellaSwag failed: {e}")
            results["HellaSwag"] = {"error": str(e)}
        
        # 5. WinoGrande
        print(f"  [{description}] WinoGrande...")
        try:
            winogrande = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
            winogrande_subset = list(winogrande.shuffle(seed=42).select(range(min(BENCHMARK_SAMPLE_SIZE, len(winogrande)))))
            
            correct = 0
            for item in winogrande_subset:
                sentence = item["sentence"]
                option1 = item["option1"]
                option2 = item["option2"]
                correct_idx = int(item["answer"]) - 1
                
                prompt = f"Fill in the blank with the correct option:\n\n{sentence}\n\n"
                prompt += f"A. {option1}\nB. {option2}\n"
                prompt += "\nAnswer with just the letter (A or B):"
                
                if evaluate_multiple_choice(model, tokenizer, prompt, [option1, option2]) == correct_idx:
                    correct += 1
            
            accuracy = correct / len(winogrande_subset)
            results["WinoGrande"] = {"accuracy": accuracy, "correct": correct, "total": len(winogrande_subset)}
            print(f"    WinoGrande: {accuracy:.2%}")
        except Exception as e:
            print(f"    WinoGrande failed: {e}")
            results["WinoGrande"] = {"error": str(e)}
        
        # Summary
        successful = [r for r in results.values() if "accuracy" in r]
        if successful:
            avg = np.mean([r["accuracy"] for r in successful])
            print(f"\n  [{description}] Average: {avg:.2%}")
        
        return results

    # ==================================================
    # Baseline Evaluation (Before Training)
    # ==================================================
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION (Before GRPO Training)")
    print("=" * 60)
    
    # Primary task evaluation
    baseline_metrics, baseline_correctness = evaluate_model(
        model, tokenizer, eval_dataset, 
        num_samples=config.EVAL_SAMPLES,
        description="Baseline (Pre-training)"
    )
    results_tracker.baseline_evaluation = baseline_metrics

    # Standard benchmarks (baseline)
    print("\n" + "=" * 60)
    print("BASELINE STANDARD BENCHMARKS")
    print("=" * 60)
    baseline_benchmark_results = run_standard_benchmarks(model, tokenizer, "Baseline")
    results_tracker.baseline_benchmark_results = baseline_benchmark_results

    # Log baseline to wandb
    if wandb_run:
        wandb.log({
            "baseline/accuracy": baseline_metrics["accuracy"],
            "baseline/macro_f1": baseline_metrics["macro_f1"],
            "baseline/brier_score": baseline_metrics.get("brier_score", 0),
            "baseline/ece": baseline_metrics.get("ece", 0),
        })
        # Log baseline benchmarks
        for bench_name, result in baseline_benchmark_results.items():
            if "accuracy" in result:
                wandb.log({f"baseline_bench/{bench_name}": result["accuracy"]})

    # ==================================================
    # GRPO Training
    # ==================================================
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        learning_rate=config.LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        num_generations=config.NUM_GENERATIONS,
        max_prompt_length=config.MAX_PROMPT_LENGTH,
        max_completion_length=config.MAX_SEQ_LENGTH - config.MAX_PROMPT_LENGTH,
        max_steps=config.MAX_STEPS,
        save_steps=config.SAVE_STEPS,
        max_grad_norm=0.1,
        num_train_epochs=1,
        
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        
        logging_steps=1,
        report_to="wandb",
        
        output_dir=config.OUTPUT_DIR,
    )

    # Custom callback for rolling evaluation
    from transformers import TrainerCallback

    class RollingEvalCallback(TrainerCallback):
        def __init__(self, eval_fn, eval_data, interval, results_tracker):
            self.eval_fn = eval_fn
            self.eval_data = eval_data
            self.interval = interval
            self.results_tracker = results_tracker
            
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step > 0 and state.global_step % self.interval == 0:
                print(f"\n--- Rolling Evaluation at Step {state.global_step} ---")
                try:
                    metrics, _ = self.eval_fn(
                        model, tokenizer, self.eval_data,
                        num_samples=min(30, len(self.eval_data)),
                        description=f"Rolling Eval (Step {state.global_step})"
                    )
                    
                    # Log to results tracker
                    self.results_tracker.log_rolling_evaluation(state.global_step, {
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "brier_score": metrics.get("brier_score"),
                        "ece": metrics.get("ece"),
                    })
                    
                    # Log to wandb
                    if wandb.run is not None:
                        wandb.log({
                            "rolling/accuracy": metrics["accuracy"],
                            "rolling/macro_f1": metrics["macro_f1"],
                            "rolling/brier_score": metrics.get("brier_score", 0),
                            "rolling/ece": metrics.get("ece", 0),
                            "rolling/step": state.global_step,
                        })
                except Exception as e:
                    print(f"Rolling evaluation failed: {e}")
                
                # Put model back in training mode
                if model is not None:
                    FastLanguageModel.for_training(model)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            format_reward_func,
            reasoning_length_reward_func,
        ],
        args=training_args,
        train_dataset=grpo_dataset,
        callbacks=[
            RollingEvalCallback(
                evaluate_model, eval_dataset, 
                config.ROLLING_EVAL_INTERVAL, results_tracker
            )
        ],
    )

    print("\n" + "=" * 60)
    print("STARTING GRPO TRAINING")
    print("=" * 60)
    trainer.train()

    # ==================================================
    # Save Checkpoints
    # ==================================================
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    model.save_pretrained(f"{config.CHECKPOINT_DIR}/lora_final")
    tokenizer.save_pretrained(f"{config.CHECKPOINT_DIR}/lora_final")
    print(f"\nLoRA weights saved to {config.CHECKPOINT_DIR}/lora_final")

    # ==================================================
    # Final Evaluation (After Training)
    # ==================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (After GRPO Training)")
    print("=" * 60)
    
    final_metrics, final_correctness = evaluate_model(
        model, tokenizer, eval_dataset,
        num_samples=config.EVAL_SAMPLES,
        description="Final (Post-training)"
    )
    results_tracker.final_evaluation = final_metrics

    # Compute statistical significance vs baseline
    if baseline_correctness and final_correctness:
        try:
            p_value = results_tracker.compute_mcnemar_test(baseline_correctness, final_correctness)
            final_metrics["mcnemar_p_value"] = p_value
            print(f"\nStatistical Significance (McNemar's test p-value): {p_value:.4f}" if p_value else "\nStatistical Significance: Unable to compute")
        except Exception as e:
            print(f"Could not compute McNemar's test: {e}")
    
    # Compute improvement
    improvement = final_metrics["accuracy"] - baseline_metrics["accuracy"]
    relative_improvement = improvement / baseline_metrics["accuracy"] if baseline_metrics["accuracy"] > 0 else 0
    
    print(f"\n--- Performance Improvement ---")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.2%} → {final_metrics['accuracy']:.2%} ({improvement:+.2%}, {relative_improvement:+.1%} relative)")
    print(f"  Macro F1: {baseline_metrics['macro_f1']:.3f} → {final_metrics['macro_f1']:.3f}")
    if baseline_metrics.get("brier_score") and final_metrics.get("brier_score"):
        brier_improvement = baseline_metrics["brier_score"] - final_metrics["brier_score"]
        print(f"  Brier Score: {baseline_metrics['brier_score']:.4f} → {final_metrics['brier_score']:.4f} ({brier_improvement:+.4f})")
    
    # Log final metrics to wandb if still active
    try:
        if wandb.run is not None:
            wandb.log({
                "final/accuracy": final_metrics["accuracy"],
                "final/macro_f1": final_metrics["macro_f1"],
                "final/brier_score": final_metrics.get("brier_score", 0),
                "final/ece": final_metrics.get("ece", 0),
                "final/improvement_absolute": improvement,
                "final/improvement_relative": relative_improvement,
            })
    except:
        pass  # wandb may have finished

    # ==================================================
    # Calculate Perplexity
    # ==================================================
    print("\n--- Perplexity Calculation ---")
    eval_texts = [item["prompt"] for item in list(eval_dataset)[:20]]
    try:
        perplexity = calculate_perplexity(model, tokenizer, eval_texts)
        print(f"  Perplexity: {perplexity:.2f}")
        final_metrics["perplexity"] = perplexity
    except Exception as e:
        print(f"  Perplexity calculation failed: {e}")

    # ==================================================
    # Test Inference Examples
    # ==================================================
    print("\n" + "=" * 60)
    print("TEST INFERENCE EXAMPLES")
    print("=" * 60)

    test_cases = [
        {
            "question": "Will the S&P 500 close above 5000 by end of Q2 2025?",
            "description": "This market resolves to Yes if the S&P 500 index closes above 5000 on any trading day before July 1, 2025."
        },
        {
            "question": "Will there be a major AI regulation bill passed in the US by 2025?",
            "description": "Resolves Yes if the US Congress passes comprehensive AI regulation legislation that is signed into law before January 1, 2026."
        }
    ]

    for i, test_case in enumerate(test_cases):
        test_prompt = format_prompt(test_case["question"], test_case["description"])
        test_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_prompt},
        ]
        test_input = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print(f"\n--- Test Case {i + 1}: {test_case['question'][:50]}... ---")
        try:
            output = generate_response(model, tokenizer, test_input)
            predicted = extract_answer(output)
            print(f"Prediction: {predicted}")
            print(f"Response:\n{output[:800]}..." if len(output) > 800 else f"Response:\n{output}")
        except Exception as e:
            print(f"Generation failed: {e}")

    # ==================================================
    # Advanced Calibration Analysis (Research Angle 1)
    # ==================================================
    print("\n" + "=" * 60)
    print("ADVANCED CALIBRATION ANALYSIS")
    print("=" * 60)
    
    # Compute advanced calibration metrics for baseline and final model
    if baseline_metrics.get("raw_data") and final_metrics.get("raw_data"):
        baseline_probs = baseline_metrics["raw_data"]["confidences"]
        baseline_gt = baseline_metrics["raw_data"]["ground_truths"]
        final_probs = final_metrics["raw_data"]["confidences"]
        final_gt = final_metrics["raw_data"]["ground_truths"]
        
        baseline_adv_cal = results_tracker.compute_advanced_calibration_metrics(baseline_probs, baseline_gt)
        final_adv_cal = results_tracker.compute_advanced_calibration_metrics(final_probs, final_gt)
        
        print("\n--- Calibration Metrics Comparison ---")
        print(f"{'Metric':<35} {'Baseline':>12} {'Final':>12} {'Change':>12}")
        print("-" * 75)
        
        for metric in ["brier_score", "brier_skill_score", "log_loss", "mce", "ace", 
                       "calibration_curve_area", "resolution", "reliability",
                       "overconfidence_rate_70", "overconfidence_rate_80", "overconfidence_rate_90"]:
            base_val = baseline_adv_cal.get(metric)
            final_val = final_adv_cal.get(metric)
            if base_val is not None and final_val is not None:
                change = final_val - base_val
                # For some metrics, lower is better
                if metric in ["brier_score", "log_loss", "mce", "ace", "calibration_curve_area", 
                              "reliability", "overconfidence_rate_70", "overconfidence_rate_80", "overconfidence_rate_90"]:
                    direction = "↓" if change < 0 else "↑"
                else:
                    direction = "↑" if change > 0 else "↓"
                print(f"{metric:<35} {base_val:>12.4f} {final_val:>12.4f} {change:>+11.4f} {direction}")
        
        # Store in results
        results_tracker.calibration_data = {
            "baseline": baseline_adv_cal,
            "final": final_adv_cal,
        }
        
        print(f"\n--- Overconfidence Analysis ---")
        print(f"  Baseline avg confidence when wrong: {baseline_adv_cal.get('avg_confidence_when_wrong', 'N/A')}")
        print(f"  Final avg confidence when wrong: {final_adv_cal.get('avg_confidence_when_wrong', 'N/A')}")
        print(f"  Baseline avg confidence when right: {baseline_adv_cal.get('avg_confidence_when_right', 'N/A')}")
        print(f"  Final avg confidence when right: {final_adv_cal.get('avg_confidence_when_right', 'N/A')}")

    # ==================================================
    # Post-Training Standard Benchmark Evaluation (Research Angle 2)
    # ==================================================
    print("\n" + "=" * 60)
    print("POST-TRAINING STANDARD BENCHMARKS")
    print("=" * 60)
    
    benchmark_results = run_standard_benchmarks(model, tokenizer, "Post-GRPO")
    results_tracker.benchmark_results = benchmark_results
    
    # Log post-training benchmarks to wandb
    try:
        if wandb.run is not None:
            for bench_name, result in benchmark_results.items():
                if "accuracy" in result:
                    wandb.log({f"final_bench/{bench_name}": result["accuracy"]})
    except:
        pass
    
    # ==================================================
    # Benchmark Comparison: Before vs After GRPO
    # ==================================================
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON: BASELINE vs POST-GRPO")
    print("=" * 60)
    print(f"{'Benchmark':<15} {'Baseline':>12} {'Post-GRPO':>12} {'Change':>12} {'Improved':>10}")
    print("-" * 65)
    
    benchmark_improvements = {}
    for bench_name in benchmark_results.keys():
        baseline_result = baseline_benchmark_results.get(bench_name, {})
        final_result = benchmark_results.get(bench_name, {})
        
        if "accuracy" in baseline_result and "accuracy" in final_result:
            baseline_acc = baseline_result["accuracy"]
            final_acc = final_result["accuracy"]
            change = final_acc - baseline_acc
            improved = "✓" if change > 0 else ("=" if change == 0 else "✗")
            
            benchmark_improvements[bench_name] = {
                "baseline": baseline_acc,
                "final": final_acc,
                "change": change,
                "improved": change > 0
            }
            
            print(f"{bench_name:<15} {baseline_acc:>11.2%} {final_acc:>11.2%} {change:>+11.2%} {improved:>10}")
        else:
            print(f"{bench_name:<15} {'ERROR':>12} {'ERROR':>12} {'-':>12} {'-':>10}")
    
    # Summary statistics
    successful_comparisons = [v for v in benchmark_improvements.values()]
    if successful_comparisons:
        avg_baseline = np.mean([v["baseline"] for v in successful_comparisons])
        avg_final = np.mean([v["final"] for v in successful_comparisons])
        avg_change = np.mean([v["change"] for v in successful_comparisons])
        num_improved = sum(1 for v in successful_comparisons if v["improved"])
        
        print("-" * 65)
        print(f"{'AVERAGE':<15} {avg_baseline:>11.2%} {avg_final:>11.2%} {avg_change:>+11.2%} {f'{num_improved}/{len(successful_comparisons)}':>10}")
        
        # Store in results tracker for paper output
        results_tracker.benchmark_comparison = {
            "per_benchmark": benchmark_improvements,
            "summary": {
                "avg_baseline": avg_baseline,
                "avg_final": avg_final,
                "avg_change": avg_change,
                "num_improved": num_improved,
                "total_benchmarks": len(successful_comparisons)
            }
        }
    
    print("\n" + "=" * 60)
    if avg_change > 0:
        print(f"TRANSFER LEARNING: POSITIVE ({num_improved}/{len(successful_comparisons)} benchmarks improved)")
        print("This supports Research Angle 2: Forecasting training transfers to general reasoning")
    elif avg_change < 0:
        print(f"TRANSFER LEARNING: MIXED ({num_improved}/{len(successful_comparisons)} benchmarks improved)")
        print("Results suggest forecasting training may be task-specific")
    else:
        print("TRANSFER LEARNING: NEUTRAL (no clear improvement or degradation)")
    print("=" * 60)

    # ==================================================
    # Generate Comprehensive Paper Output
    # ==================================================
    print("\n" + "=" * 60)
    print("GENERATING PAPER OUTPUT")
    print("=" * 60)

    # Save comprehensive results
    paper_output = results_tracker.generate_paper_output()
    
    # Remove raw data for cleaner JSON (can be large)
    paper_output_clean = paper_output.copy()
    if "final_evaluation" in paper_output_clean and paper_output_clean["final_evaluation"]:
        paper_output_clean["final_evaluation"] = {
            k: v for k, v in paper_output_clean["final_evaluation"].items() 
            if k != "raw_data"
        }
    if "baseline_evaluation" in paper_output_clean and paper_output_clean["baseline_evaluation"]:
        paper_output_clean["baseline_evaluation"] = {
            k: v for k, v in paper_output_clean["baseline_evaluation"].items()
            if k != "raw_data"
        }

    # Save JSON results
    results_json_path = os.path.join(config.RESULTS_DIR, "experiment_results.json")
    with open(results_json_path, "w") as f:
        json.dump(paper_output_clean, f, indent=2, default=str)
    print(f"Results saved to: {results_json_path}")

    # Generate prompts for paper sections
    methods_prompt = results_tracker.generate_methods_section_prompt()
    results_prompt = results_tracker.generate_results_section_prompt()

    # Save prompts
    methods_prompt_path = os.path.join(config.RESULTS_DIR, "methods_section_prompt.txt")
    results_prompt_path = os.path.join(config.RESULTS_DIR, "results_section_prompt.txt")
    
    with open(methods_prompt_path, "w") as f:
        f.write(methods_prompt)
    print(f"Methods section prompt saved to: {methods_prompt_path}")
    
    with open(results_prompt_path, "w") as f:
        f.write(results_prompt)
    print(f"Results section prompt saved to: {results_prompt_path}")

    # ==================================================
    # Print Summary for Paper
    # ==================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY FOR PAPER")
    print("=" * 60)
    
    # Build calibration comparison for summary
    cal_baseline = results_tracker.calibration_data.get('baseline', {}) if results_tracker.calibration_data else {}
    cal_final = results_tracker.calibration_data.get('final', {}) if results_tracker.calibration_data else {}
    
    # Build benchmark comparison summary
    bench_comparison = ""
    bench_comparison_data = getattr(results_tracker, 'benchmark_comparison', {})
    per_bench = bench_comparison_data.get('per_benchmark', {})
    
    for name in benchmark_results.keys():
        if name in per_bench:
            data = per_bench[name]
            change_str = f"{data['change']:+.2%}"
            improved = "✓" if data['improved'] else "✗"
            bench_comparison += f"| {name} | {data['baseline']:.2%} | {data['final']:.2%} | {change_str} | {improved} |\n"
        elif "accuracy" in benchmark_results.get(name, {}):
            bench_comparison += f"| {name} | N/A | {benchmark_results[name]['accuracy']:.2%} | N/A | - |\n"
    
    summary = f"""
## GRPO for Prediction Market Forecasting - Experiment Summary

### Research Questions
1. **Research Angle 1**: Can RL (GRPO) teach LLMs epistemic calibration?
2. **Research Angle 2**: Does forecasting training transfer to general reasoning?

### Experimental Setup
- **Model**: {config.MODEL_NAME}
- **Fine-tuning**: LoRA (rank={config.LORA_RANK}) with GRPO
- **Training Steps**: {config.MAX_STEPS}
- **Learning Rate**: {config.LEARNING_RATE}
- **Dataset**: {config.DATASET_NAME}
- **Evaluation Samples**: {config.EVAL_SAMPLES}

---

### PRIMARY TASK: Prediction Market Forecasting

| Metric | Baseline | After GRPO | Improvement |
|--------|----------|------------|-------------|
| Accuracy | {baseline_metrics['accuracy']:.2%} | {final_metrics['accuracy']:.2%} | {improvement:+.2%} |
| Macro F1 | {baseline_metrics['macro_f1']:.3f} | {final_metrics['macro_f1']:.3f} | {final_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.3f} |
| Brier Score | {baseline_metrics.get('brier_score', 'N/A'):.4f if baseline_metrics.get('brier_score') else 'N/A'} | {final_metrics.get('brier_score', 'N/A'):.4f if final_metrics.get('brier_score') else 'N/A'} | {baseline_metrics.get('brier_score', 0) - final_metrics.get('brier_score', 0):+.4f if baseline_metrics.get('brier_score') and final_metrics.get('brier_score') else 'N/A'} |
| ECE | {baseline_metrics.get('ece', 'N/A'):.4f if baseline_metrics.get('ece') else 'N/A'} | {final_metrics.get('ece', 'N/A'):.4f if final_metrics.get('ece') else 'N/A'} | {baseline_metrics.get('ece', 0) - final_metrics.get('ece', 0):+.4f if baseline_metrics.get('ece') and final_metrics.get('ece') else 'N/A'} |

**Statistical Significance**
- McNemar's Test p-value: {final_metrics.get('mcnemar_p_value', 'N/A')}
- 95% CI for Final Accuracy: [{final_metrics.get('accuracy_ci_lower', 0):.2%}, {final_metrics.get('accuracy_ci_upper', 0):.2%}]

---

### RESEARCH ANGLE 1: Calibration Analysis

| Calibration Metric | Baseline | Final | Change (↓ better) |
|--------------------|----------|-------|-------------------|
| Brier Score | {cal_baseline.get('brier_score', 'N/A'):.4f if cal_baseline.get('brier_score') else 'N/A'} | {cal_final.get('brier_score', 'N/A'):.4f if cal_final.get('brier_score') else 'N/A'} | {(cal_baseline.get('brier_score', 0) - cal_final.get('brier_score', 0)):+.4f if cal_baseline.get('brier_score') and cal_final.get('brier_score') else 'N/A'} |
| Log Loss | {cal_baseline.get('log_loss', 'N/A'):.4f if cal_baseline.get('log_loss') else 'N/A'} | {cal_final.get('log_loss', 'N/A'):.4f if cal_final.get('log_loss') else 'N/A'} | {(cal_baseline.get('log_loss', 0) - cal_final.get('log_loss', 0)):+.4f if cal_baseline.get('log_loss') and cal_final.get('log_loss') else 'N/A'} |
| Max Calibration Error | {cal_baseline.get('mce', 'N/A'):.4f if cal_baseline.get('mce') else 'N/A'} | {cal_final.get('mce', 'N/A'):.4f if cal_final.get('mce') else 'N/A'} | {(cal_baseline.get('mce', 0) - cal_final.get('mce', 0)):+.4f if cal_baseline.get('mce') and cal_final.get('mce') else 'N/A'} |
| Overconf Rate (80%) | {cal_baseline.get('overconfidence_rate_80', 'N/A'):.2% if cal_baseline.get('overconfidence_rate_80') else 'N/A'} | {cal_final.get('overconfidence_rate_80', 'N/A'):.2% if cal_final.get('overconfidence_rate_80') else 'N/A'} | - |

**Key Finding**: {('Calibration IMPROVED - supports Research Angle 1' if cal_final.get('brier_score', 1) < cal_baseline.get('brier_score', 0) else 'Calibration needs further analysis')}

---

### RESEARCH ANGLE 2: Standard Benchmarks (Transfer Learning)

| Benchmark | Baseline | Post-GRPO | Change | Improved |
|-----------|----------|-----------|--------|----------|
{bench_comparison}
*Benchmarks: TruthfulQA (truthfulness), ARC-Easy (science reasoning), MMLU (general knowledge), HellaSwag (commonsense), WinoGrande (coreference)*

**Summary**: {bench_comparison_data.get('summary', {}).get('num_improved', 0)}/{bench_comparison_data.get('summary', {}).get('total_benchmarks', 0)} benchmarks improved, avg change: {bench_comparison_data.get('summary', {}).get('avg_change', 0):+.2%}

---

### Training Dynamics
- Rolling evaluations logged every {config.ROLLING_EVAL_INTERVAL} steps
- Total rolling evaluations: {len(results_tracker.rolling_evaluations)}

### Files Generated
- Experiment Results (JSON): {results_json_path}
- Methods Section Prompt: {methods_prompt_path}
- Results Section Prompt: {results_prompt_path}
- Model Checkpoint: {config.CHECKPOINT_DIR}/lora_final

### Suggested Paper Title
"Teaching LLMs Epistemic Calibration: GRPO Fine-tuning on Prediction Markets"

### Key Claims (if results support)
1. GRPO training on prediction markets improves model calibration (Brier Score, ECE)
2. Forecasting training may transfer to related reasoning tasks
3. RL can teach epistemic humility to language models
"""
    
    print(summary)
    
    # Save summary
    summary_path = os.path.join(config.RESULTS_DIR, "experiment_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")

    # ==================================================
    # Cleanup
    # ==================================================
    try:
        wandb.finish()
    except:
        pass

    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"All outputs saved to: {output_dir}")
    
    return paper_output
