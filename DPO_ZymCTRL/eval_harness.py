"""
ZymCTRL Evaluation Harness
=========================

This module provides evaluation tools for ZymCTRL models, specifically analyzing:
1. Base model performance (no control tags)
2. Controllability (effectiveness of stability tags)
3. Model preservation (comparing finetuned vs base model)

Usage
-----
1. Base Model Performance:
   Evaluates the base model's generation capabilities without control tags
   We will evaluate the 
   ```
   Command: python eval_harness.py \
    --model_path "AI4PD/ZymCTRL" \
    --eval_type performance \
    --output_dir "base_results"
   ```
}

2. Controllability Analysis:
   Tests how well stability tags (high/medium/low) affect generation. 
   We will evaluate the distribution of stabilities/preplexities for each stability tag

    Command: python eval_harness.py \
       --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_agi/epoch=0-val_loss=0.46.ckpt" \
       --eval_type controllability \
       --output_dir "ctrl_results"


3. Model Preservation:
   Compares finetuned model to base model (requires running both). We use NO control tags for this eval
   ```
   # First run base model
    Command: python eval_harness.py \
       --model_path "AI4PD/ZymCTRL" \
       --eval_type performance \
       --output_dir "comparison/base"

   # Then run finetuned model
   python eval_harness.py \
       --model_path "path/to/checkpoint.ckpt" \
       --eval_type performance \
       --output_dir "comparison/finetuned"
   ```

Arguments
---------
--model_path: Path to model or checkpoint
--eval_type: Type of evaluation [performance|controllability|all]
--num_samples: Number of sequences to generate (default: 100)
--batch_size: Batch size for generation (default: 10)
--ec_label: EC number to use (default: 4.2.1.1)
--output_dir: Directory to save results

Output Format
------------
Results are saved as JSON files containing:
- Generated sequences
- Perplexity scores
- Stability scores
- Statistical comparisons (for controllability)

Additional visualization tools are available in eval_viz.py
"""

import os
import math
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
import Levenshtein

from stability import stability_score_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<start>", "<end>", "<|endoftext|>", "<pad>", " ", "<sep>"]
STABILITY_TAGS = ["<stability=high>", "<stability=medium>", "<stability=low>"]

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


@dataclass
class GenerationConfig:
    """Configuration for sequence generation."""
    max_length: int = 400
    min_length: int = 10
    top_k: int = 9
    repetition_penalty: float = 1.2
    do_sample: bool = True
    num_return_sequences: int = 1
    temperature: float = 1
class BaseEvaluator:
    """Base evaluation functionality shared by both evaluator types."""
    
    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        num_samples: int = 100,
        batch_size: int = 10,
        ec_label: str = "4.2.1.1",
        brenda_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.ec_label = ec_label
        
        # Load tokenizer and model
        self._setup_tokenizer()
        self._setup_model()
        
    def _setup_tokenizer(self) -> None:
        """Initialize and configure tokenizer."""
        if self.model_path.endswith(".ckpt"):
            self.tokenizer = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
        # Add stability tags to vocabulary
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": STABILITY_TAGS}
        )
        if added:
            logger.info("Added %d new special tokens", added)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _setup_model(self) -> None:
        """Initialize and configure model."""
        base = (
            GPT2LMHeadModel.from_pretrained("AI4PD/ZymCTRL").to(self.device)
            if self.model_path.endswith(".ckpt")
            else GPT2LMHeadModel.from_pretrained(self.model_path).to(self.device)
        )
        
        # Resize embeddings for new tokens
        base.resize_token_embeddings(len(self.tokenizer))
        
        # Load checkpoint if needed
        if self.model_path.endswith(".ckpt"):
            ckpt = torch.load(self.model_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            missing, unexpected = base.load_state_dict(cleaned_state_dict, strict=False)
            if missing or unexpected:
                logger.warning("Missing keys: %s | Unexpected keys: %s", missing, unexpected)
        
        self.model = base.to(self.device).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def _build_prompt(self, stability_tag: Optional[str] = None) -> str:
        """Create input prompt with optional stability tag."""
        tag = f"<stability={stability_tag}>" if stability_tag else ""
        return f"{self.ec_label}{tag}<sep><start>"
    
    @torch.no_grad()
    def generate_sequences(
        self,
        stability_tag: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate sequences for given stability tag."""
        if config is None:
            config = GenerationConfig()
            
        prompt_str = self._build_prompt(stability_tag)
        prompt_batch = self.tokenizer(
            [prompt_str] * self.batch_size,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        seqs: List[str] = []
        batches = math.ceil(self.num_samples / self.batch_size)
        
        for _ in tqdm(range(batches), desc=f"Generate‑{stability_tag or 'baseline'}"):
            outputs = self.model.generate(
                **prompt_batch,
                max_length=config.max_length,
                min_length=config.min_length,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                num_return_sequences=1, # needs to be since our prompt batch is already batched!
                temperature=config.temperature,
            )
            
            for seq_ids in outputs:
                text = self.tokenizer.decode(seq_ids, skip_special_tokens=False)
                
                # Clean up sequence
                clean = text.replace(self.ec_label, "")

                # print("Length of sequence: ", len(seq_ids))
                if stability_tag:
                    clean = clean.replace(f"<stability={stability_tag}>", "")
                clean = self._strip_special(clean).strip()
                
                if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in clean):
                    seqs.append(clean)
                    
            if len(seqs) >= self.num_samples:
                break
                
        return seqs[:self.num_samples]
    
    @torch.no_grad()
    def calculate_perplexity(self, sequences: Sequence[str]) -> List[float]:
        """Calculate perplexity for a list of sequences."""
        perplexities = []
        for seq in sequences:
            ids = self.tokenizer(seq, return_tensors="pt").input_ids.to(self.device)
            loss = self.model(ids, labels=ids).loss
            perplexities.append(math.exp(loss.item()))
        return perplexities
    
    def calculate_stability(self, sequences: Sequence[str]) -> List[Dict[str, float]]:
        """Calculate stability scores for sequences in batches."""
        scores = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            batch_scores = stability_score_batch(batch)
            scores.extend([{"raw_if": raw_if, "dg": dg, "plddt": plddt} for raw_if, dg, plddt in batch_scores])
        return scores
    
    @staticmethod
    def _strip_special(seq: str) -> str:
        """Remove special tokens and training markup."""
        *_, core = seq.split("<sep>")
        for token in SPECIAL_TOKENS:
            core = core.replace(token, "")
        return core
    
    def _load_brenda(self) -> List[str]:
        df = pd.read_csv(self.brenda_path)
        filtered_df = df[df['EC_NUMBER'] == self.ec_label]
        return filtered_df.tolist()

class ModelPerformanceEvaluator(BaseEvaluator):
    """Evaluates model performance relative to baseline."""
    
    def evaluate(self) -> Dict[str, Any]:
        """Run performance evaluation."""
        sequences = self.generate_sequences()
        stability_metrics = self.calculate_stability(sequences)
        
        return {
            "sequences": sequences,
            "metrics": {
                "perplexity": self.calculate_perplexity(sequences),
                "stability": {
                    "raw_if": [m["raw_if"] for m in stability_metrics],
                    "dg": [m["dg"] for m in stability_metrics],
                    "plddt": [m["plddt"] for m in stability_metrics]
                }
            }
        }

class ControllabilityEvaluator(BaseEvaluator):
    """Evaluates effectiveness of stability tags."""
    
    def evaluate(self) -> Dict[str, Any]:
        """Run controllability evaluation."""
        results = {}
        
        # Generate and evaluate sequences for each stability level
        for tag in ["high", "medium", "low"]:
            sequences = self.generate_sequences(stability_tag=tag)
            stability_metrics = self.calculate_stability(sequences)
            
            results[tag] = {
                "sequences": sequences,
                "metrics": {
                    "perplexity": self.calculate_perplexity(sequences),
                    "stability": {
                        "raw_if": [m["raw_if"] for m in stability_metrics],
                        "dg": [m["dg"] for m in stability_metrics],
                        "plddt": [m["plddt"] for m in stability_metrics]
                    }
                }
            }
            
        return results

class MembershipEvaluator(BaseEvaluator):
    """Evaluates membership of generated sequences in BRENDA database."""

    @staticmethod
    def membership_score(brenda_sequences, target_sequences):
        scores = []

        for target_seq in target_sequences:
            min_score = None
            for brenda_seq in brenda_sequences:
                score = Levenshtein.distance(target_seq, brenda_seq)
                if min_score is None or score < min_score:
                    min_score = score
            scores.append(min_score)

        return scores

    def evaluate(self) -> Dict[str, Any]:
        """Run membership evaluation."""
        brenda_sequences = self._load_brenda()

        results = {}

        for tag in ["high", "medium", "low"]:
            sequences = self.generate_sequences(stability_tag=tag)
            scores = self.membership_score(brenda_sequences, sequences)

            results[tag] = {
                "sequences": sequences,
                "scores": scores
            }

        return results

class EvaluationRunner:
    """High-level interface for running evaluations."""
    
    def __init__(
        self,
        model_path: str,
        num_samples: int = 100,
        batch_size: int = 10,
        ec_label: str = "4.2.1.1",
    ) -> None:
        self.model_path = model_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.ec_label = ec_label
        
    def run_performance(self) -> Dict[str, Any]:
        """Run model performance evaluation."""
        evaluator = ModelPerformanceEvaluator(
            self.model_path,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            ec_label=self.ec_label,
        )
        return evaluator.evaluate()
    
    def run_controllability(self) -> Dict[str, Any]:
        """Run controllability evaluation."""
        evaluator = ControllabilityEvaluator(
            self.model_path,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            ec_label=self.ec_label,
        )
        return evaluator.evaluate()
    
    def run_membership(self) -> Dict[str, Any]:
        """Run membership evaluation."""
        evaluator = MembershipEvaluator(
            self.model_path,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
        )

    def run_all(self) -> Dict[str, Any]:
        """Run both performance and controllability evaluations."""
        return {
            "performance": self.run_performance(),
            "controllability": self.run_controllability()
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ZymCTRL model performance and controllability")
    parser.add_argument("--model_path", required=True, help="Path to model or checkpoint")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of sequences to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--ec_label", type=str, default="4.2.1.1", help="EC label to use")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument(
        "--eval_type",
        choices=["all", "performance", "controllability"],
        default="all",
        help="Type of evaluation to run",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    runner = EvaluationRunner(
        model_path=args.model_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ec_label=args.ec_label,
    )
    
    if args.eval_type == "performance":
        results = runner.run_performance()
    elif args.eval_type == "controllability":
        results = runner.run_controllability()
    else:
        results = runner.run_all()
    
    # Save results
    output_path = os.path.join(args.output_dir, f"{args.eval_type}_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
