import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import logging
from tqdm import tqdm
from typing import Optional

from transformers import AutoTokenizer, GPT2LMHeadModel
from stability import stability_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

special_tokens = ['<start>', '<end>', '<|endoftext|>', '<pad>', ' ', '<sep>']

def remove_characters(sequence, char_list):
    '''
    Removes special tokens used during training.
    '''
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq



class ControllabilityEvaluator:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_samples: int = 100,
        ec_label: str = "4.2.1.1",
        batch_size: int = 10,
    ):
        """Initialize the controllability evaluator."""
        self.model_path = model_path
        self.device = device
        self.num_samples = num_samples
        self.ec_label = ec_label
        self.batch_size = batch_size
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        
        # Check if path is a checkpoint file
        if model_path.endswith('.ckpt'):
            # Load just the tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL')
            
            # Initialize a fresh model without pretrained weights
            config = GPT2LMHeadModel.from_pretrained('AI4PD/ZymCTRL').config
            base_model = GPT2LMHeadModel(config)
            
            # Add special tokens for stability tags and sequence formatting
            special_tokens_dict = {
                "additional_special_tokens": [
                    "<stability=high>",
                    "<stability=medium>",
                    "<stability=low>"
                ]
            }
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            logger.info(f"Added {num_added_toks} special tokens to tokenizer")
            
            # Resize model embeddings to account for new special tokens
            base_model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized model embeddings to {len(self.tokenizer)}")
            
            # Load state dict from checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            # Extract model state dict - handle both Lightning and regular checkpoints
            if 'state_dict' in checkpoint:
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            else:
                state_dict = checkpoint
            
            # Load the state dict
            base_model.load_state_dict(state_dict)
            self.model = base_model.to(device)
            logger.info(f"Loaded checkpoint from {model_path}")
        else:
            # Regular model loading
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        
        # Set up padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set padding token to EOS token")
        
        # Make sure model knows about padding token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Define stability levels
        self.stability_levels = ["high", "medium", "low", None]
    
    def _format_sequence(self, ec_label: str, stability_level: Optional[str] = None) -> str:
        """Format sequence according to original ZymCTRL paper format"""
        if stability_level is not None:
            # Add stability control tag right after EC label
            return f"{ec_label}<stability={stability_level}><sep><start>"
        return f"{ec_label}<sep><start>"
        
    def _generate_sequences(self, stability_level: str = None) -> list:
        """Generate sequences for a given stability level."""
        sequences = []

        prompt = self._format_sequence(self.ec_label, stability_level)
            
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate sequences in batches
        num_batches = self.num_samples // self.batch_size
        
        for _ in tqdm(range(num_batches), desc=f"Generating {stability_level} sequences"):
            try:
                # Generate using same parameters as generate_training_data.pyc
                outputs = self.model.generate(
                    input_ids, 
                    top_k=9, 
                    repetition_penalty=1.2,
                    max_length=1024,
                    min_length=10,
                    eos_token_id=1,
                    pad_token_id=0,
                    do_sample=True,
                    num_return_sequences=self.batch_size
                )

                # Filter sequences same way as generate_training_data.py
                new_outputs = [output for output in outputs if output[-1] == 0 or output[-1] == 1]
                
                if not new_outputs:
                    logger.warning("No properly terminated sequences in batch")
                    continue

                # Process valid sequences
                for output in new_outputs:
                    seq = self.tokenizer.decode(output)
                    # Clean sequence same way as generate_training_data.py
                    seq = seq.replace(self.ec_label, "").strip()
                    if stability_level:
                        seq = seq.replace(f"<stability={stability_level}>", "").strip()
                        seq = remove_characters(seq, special_tokens)
                    # Only keep sequences with valid amino acids
                    if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
                        sequences.append(seq)

                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                logger.error(f"Error during generation: {str(e)}")
                continue

        return sequences[:self.num_samples]
        
        
    def evaluate(self):
        """Run controllability evaluation."""
        results = {
            "sequences": {},
            "stability_scores": {},
            "statistics": {},
        }
        
        # 1. Generate sequences and calculate stability for each level
        for level in self.stability_levels:
            logger.info(f"Generating sequences for stability level: {level}")
            sequences = self._generate_sequences(level)
            logger.info(f"Generated {len(sequences)} sequences for stability level: {level}")
            results["sequences"][level] = sequences
            
            # Calculate stability scores
            logger.info(f"Calculating stability scores for {level}")
            scores = stability_score(sequences)
            logger.info(f"Number of stability scores evaluated: {len(scores)}")
            assert len(sequences) == len(scores), "Num of stabilty scores should be same as generated sequences"
            results["stability_scores"][level] = [score[1] for score in scores]  # Use deltaG values
            
            # Calculate basic statistics
            scores_array = np.array(results["stability_scores"][level])
            results["statistics"][level] = {
                "mean": np.mean(scores_array),
                "std": np.std(scores_array),
                "min": np.min(scores_array),
                "max": np.max(scores_array)
            }
        
        # 2. Calculate statistical significance between pairs
        from scipy import stats
        results["comparisons"] = {}
        for level1 in self.stability_levels:
            for level2 in self.stability_levels:
                if level1 < level2:  # Only do each pair once
                    scores1 = results["stability_scores"][level1]
                    scores2 = results["stability_scores"][level2]
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    mean_diff = np.mean(scores1) - np.mean(scores2)
                    results["comparisons"][f"{level1}_vs_{level2}"] = {
                        "mean_difference": mean_diff,
                        "p_value": p_value
                    }
                    
        return results
        
    def plot_results(self, results, output_dir="eval_results"):
        """Plot evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot stability score distributions
        plt.figure(figsize=(10, 6))
        for level in self.stability_levels:
            sns.kdeplot(results["stability_scores"][level], label=f"{level} (mean={np.mean(results['stability_scores'][level]):.2f})")
        plt.title("Stability Score Distributions")
        plt.xlabel("ΔG (kcal/mol)")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "stability_distributions.png"))
        plt.close()
        
        # Save numerical results
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write("Controllability Evaluation Results\n")
            f.write("================================\n\n")
            
            # Stability statistics
            f.write("Stability Statistics:\n")
            f.write("-----------------\n")
            for level in self.stability_levels:
                stats = results["statistics"][level]
                f.write(f"{level}:\n")
                f.write(f"  Mean ΔG: {stats['mean']:.2f}\n")
                f.write(f"  Std ΔG: {stats['std']:.2f}\n")
                f.write(f"  Min ΔG: {stats['min']:.2f}\n")
                f.write(f"  Max ΔG: {stats['max']:.2f}\n\n")
                
            # Statistical comparisons
            f.write("\nStability Level Comparisons:\n")
            f.write("-------------------------\n")
            for comparison, stats in results["comparisons"].items():
                f.write(f"{comparison}:\n")
                f.write(f"  Mean difference: {stats['mean_difference']:.2f} kcal/mol\n")
                f.write(f"  P-value: {stats['p_value']:.4f}\n")
                f.write(f"  Significant: {'Yes' if stats['p_value'] < 0.05 else 'No'}\n\n")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model controllability')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model or checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of sequences to generate per stability level')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Batch size for generation')
    parser.add_argument('--ec_label', type=str, default="4.2.1.1",
                      help='EC number to use')
    parser.add_argument('--output_dir', type=str, default="eval_results",
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ControllabilityEvaluator(
        model_path=args.model_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ec_label=args.ec_label
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Plot and save results
    evaluator.plot_results(results, args.output_dir) 