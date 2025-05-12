import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import logging
from typing import Optional
import numpy as np
from Bio.Align import substitution_matrices

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_sequence_identity(seq1: str, seq2: str) -> float:
    """Calculate sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        return 0.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)

def calculate_blosum62_score(seq1: str, seq2: str) -> float:
    """Calculate average BLOSUM62 score per residue between two sequences."""
    if len(seq1) != len(seq2):
        return float('-inf')
    
    blosum62 = substitution_matrices.load("BLOSUM62")
    total_score = 0
    valid_positions = 0
    
    for a, b in zip(seq1, seq2):
        try:
            score = blosum62.get((a, b), blosum62.get((b, a), 0))
            total_score += score
            valid_positions += 1
        except KeyError:
            continue
    
    return total_score / valid_positions if valid_positions > 0 else float('-inf')

def filter_valid_pairs(
    sorted_data: pd.DataFrame,
    top_indices: list,
    bottom_indices: list,
    stability_threshold: float,
    min_sequence_identity: float,
    min_blosum62_score: float
) -> list:
    """
    Filter pairs of sequences based on stability difference, sequence identity, and BLOSUM62 score.
    
    Args:
        sorted_data: DataFrame containing sequences and their stability scores
        top_indices: Indices of most stable sequences
        bottom_indices: Indices of least stable sequences
        stability_threshold: Minimum stability difference required
        min_sequence_identity: Minimum sequence identity required (0-1)
        min_blosum62_score: Minimum BLOSUM62 score per residue required
        
    Returns:
        List of valid pairs (i, j) where i is from top_indices and j is from bottom_indices
    """
    valid_pairs = []
    for i in top_indices:
        for j in bottom_indices:
            # Check stability difference
            diff = abs(sorted_data.iloc[j]['stability_score'] - sorted_data.iloc[i]['stability_score'])
            if diff < stability_threshold:
                continue
            
            # Get sequences
            seq1 = sorted_data.iloc[i]['sequence']
            seq2 = sorted_data.iloc[j]['sequence']
            
            # Check sequence identity
            identity = calculate_sequence_identity(seq1, seq2)
            
            if identity < min_sequence_identity:
                continue
            
            
            # Check BLOSUM62 score
            blosum_score = calculate_blosum62_score(seq1, seq2)
            if blosum_score < min_blosum62_score:
                continue
            
            valid_pairs.append((i, j))
    
    return valid_pairs

# Underlying dataset
class ZymCTRLDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        training_mode: str = "sft",
        stability_threshold: float = 0,
        type: str = "train"
    ):
        self.tokenizer = tokenizer
        logger.info(f"Initializing ZymCTRLDataset in {training_mode} mode")
        logger.info(f"Loading data from {data_path}")

        # Set up padding token if not set. 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set padding token to EOS token")

        self.type = type
        self.max_length = max_length
        self.training_mode = training_mode
        self.stability_threshold = stability_threshold
        logger.info(f"Using stability threshold: {stability_threshold}")
        
        # Load data
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} sequences")
        
        # Log stability score distribution
        logger.info(f"Stability score range: {self.data['stability_score'].min():.2f} to {self.data['stability_score'].max():.2f}")
        logger.info(f"Stability score mean: {self.data['stability_score'].mean():.2f}")
        logger.info(f"Stability score std: {self.data['stability_score'].std():.2f}")
        
        # Verify required columns exist
        assert "sequence" in self.data.columns, "sequence column required"
        assert "stability_score" in self.data.columns, "stability_score column required"
        assert "ec_label" in self.data.columns, "ec_label column required"

    def _format_sequence(self, ec_label: str, sequence: str, stability_level: Optional[str] = None) -> str:
        """Format sequence according to original ZymCTRL paper format"""
        if stability_level is not None:
            # Add stability control tag right after EC label
            return f"{ec_label}<stability={stability_level}><sep><start>{sequence}<end><|endoftext|>"
        return f"{ec_label}<sep><start>{sequence}<end><|endoftext|>"

    # Interfaces that child classes need to define
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()
        
# SFT layer
class ZymCTRLSFTDataset(ZymCTRLDataset):
    def __init__(
        self,
        split_percent=.33,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger.info(f"Initializing SFT ZymCTRLDataset.")

        # Process underlying data for DPO!
        # tldr; sort and pair 

        self.split_percent = split_percent

        # Sort data by stability score - ascending for stability (lower = more stable)
        sorted_data = self.data.sort_values('stability_score', ascending=True)
        n_samples = len(sorted_data)
        n_subset = int(split_percent * n_samples)
        logger.info(f"Creating subsets with {n_subset} samples each (15% of total)")

        # Select top 15% (most stable), middle 15%, and bottom 15% (least stable)
        top_indices = list(range(n_subset))  # Most stable (lowest deltaG)
        mid_start = n_samples // 2 - n_subset // 2
        middle_indices = list(range(mid_start, mid_start + n_subset))
        bottom_indices = list(range(n_samples - n_subset, n_samples))  # Least stable (highest deltaG)

        # Log stability ranges for each group
        logger.info("Stability ranges for each group:")
        logger.info(f"High stability (lowest deltaG): {sorted_data.iloc[top_indices]['stability_score'].min():.2f} to {sorted_data.iloc[top_indices]['stability_score'].max():.2f}")
        logger.info(f"Medium stability: {sorted_data.iloc[middle_indices]['stability_score'].min():.2f} to {sorted_data.iloc[middle_indices]['stability_score'].max():.2f}")
        logger.info(f"Low stability (highest deltaG): {sorted_data.iloc[bottom_indices]['stability_score'].min():.2f} to {sorted_data.iloc[bottom_indices]['stability_score'].max():.2f}")

        # Store the indices and their corresponding stability levels
        self.sample_indices = []
        for idx in top_indices:
            self.sample_indices.append((idx, 'high'))  # Most stable = high stability
        for idx in middle_indices:
            self.sample_indices.append((idx, 'medium'))
        for idx in bottom_indices:
            self.sample_indices.append((idx, 'low'))  # Least stable = low stability
            
        # Store sorted data for easy access during training
        self.sorted_data = sorted_data
        logger.info(f"Created {len(self.sample_indices)} total samples")
        
        # Count and log the number of samples for each stability level
        high_count = sum(1 for _, level in self.sample_indices if level == 'high')
        medium_count = sum(1 for _, level in self.sample_indices if level == 'medium')
        low_count = sum(1 for _, level in self.sample_indices if level == 'low')
        logger.info(f"Number of samples per stability level:")
        logger.info(f"High stability: {high_count}")
        logger.info(f"Medium stability: {medium_count}")
        logger.info(f"Low stability: {low_count}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        # Get the sample index and stability level
        data_idx, stability_level = self.sample_indices[idx]
        sample = self.sorted_data.iloc[data_idx]
        
        # Log every 1000th sample for monitoring
        if idx % 1000 == 0:
            logger.debug(f"== SFT Internal logging call ==")
            logger.debug(f"SFT Sample {idx}:")
            logger.debug(f"Stability level: {stability_level}")
            logger.debug(f"Stability score: {sample['stability_score']:.2f}")
            logger.debug(f"== SFT Internal logging end ==")
        
        # Construct prompt using original ZymCTRL format
        # stability_tag = None
        # if self.type == "train":
        stability_tag = sample['stability_label']
        
        prompt = self._format_sequence(sample['ec_label'], sample['sequence'], stability_tag)
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # NOTE: This attention mask is NOT causal. It only masks out the padding tokens.
        # However, GPT2LMHeadModel automatically uses a causal attention mask. So we are chilling

        # Return only the necessary fields for the model
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "stability_score": sample["stability_score"],
            "perplexity": sample["perplexity"]
        } 

# DPO layer
class ZymCTRLDPODataset(ZymCTRLDataset):
    def __init__(
        self,
        min_sequence_identity: float = 0.05,  # Minimum sequence identity (90%)
        min_blosum62_score: float = -1.0,    # Minimum BLOSUM62 score per residue
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_sequence_identity = min_sequence_identity
        self.min_blosum62_score = min_blosum62_score

        logger.info(f"Initializing DPO ZymCTRLDataset.")
        logger.info(f"Using sequence identity threshold: {min_sequence_identity}")
        logger.info(f"Using BLOSUM62 score threshold: {min_blosum62_score}")
        
        # ===================== Pair Construction Logic =====================
        # The goal is to create pairs of sequences with clear stability differences
        # for training the DPO (Direct Preference Optimization) model.
        #
        # Process:
        # 1. Sort all sequences by stability score (ascending, so most stable first — i.e. have lowest deltaG)
        # 2. Take the top 15% most stable sequences and bottom 15% least stable sequences
        # 3. Create pairs by matching each sequence from top 15% with each from bottom 15%
        # 4. Filter pairs to ensure they meet:
        #    - Minimum stability difference threshold
        #    - Minimum sequence identity (90%)
        #    - Minimum BLOSUM62 score per residue (-1)
        # 5. Randomly shuffle valid pairs
        # 6. Split pairs into two equal groups:
        #    - First half: prefer stable (chosen=stable, rejected=unstable)
        #    - Second half: prefer unstable (chosen=unstable, rejected=stable)
        # ==============================================================

        # Sort data by stability score for easier pairing
        sorted_data = self.data.sort_values('stability_score', ascending=True)
        n_samples = len(sorted_data)
        
        # Select top 15% (most stable) and bottom 15% least stable sequences
        n_subset = int(n_samples * 0.25)
        top_indices = list(range(n_subset))  # Most stable (lowest deltaG)
        bottom_indices = list(range(n_samples - n_subset, n_samples))  # Least stable (highest deltaG)
        
        logger.info(f"Using top and bottom {n_subset} samples each (15% of total)")
        logger.info(f"Stability ranges:")
        logger.info(f"Top (most stable): {sorted_data.iloc[top_indices]['stability_score'].min():.2f} to {sorted_data.iloc[top_indices]['stability_score'].max():.2f}")
        logger.info(f"Bottom (least stable): {sorted_data.iloc[bottom_indices]['stability_score'].min():.2f} to {sorted_data.iloc[bottom_indices]['stability_score'].max():.2f}")
        
        # Create pairs between top and bottom groups with robust filtering
        valid_pairs = filter_valid_pairs(
            sorted_data=sorted_data,
            top_indices=top_indices,
            bottom_indices=bottom_indices,
            stability_threshold=self.stability_threshold,
            min_sequence_identity=self.min_sequence_identity,
            min_blosum62_score=self.min_blosum62_score
        )
        
        if len(valid_pairs) == 0:
            raise ValueError(
                f"No valid pairs found with:\n"
                f"- Stability difference >= {self.stability_threshold}\n"
                f"- Sequence identity >= {self.min_sequence_identity}\n"
                f"- BLOSUM62 score >= {self.min_blosum62_score}\n"
                f"Try lowering the thresholds. "
                f"Min stability difference: {sorted_data['stability_score'].max() - sorted_data['stability_score'].min():.2f}"
            )
        
        logger.info(f"Found {len(valid_pairs)} valid pairs meeting all criteria")
        
        # Log some statistics about the filtered pairs
        if len(valid_pairs) > 0:
            sample_pairs = random.sample(valid_pairs, min(5, len(valid_pairs)))
            logger.info("Sample pair statistics:")
            for i, j in sample_pairs:
                seq1 = sorted_data.iloc[i]['sequence']
                seq2 = sorted_data.iloc[j]['sequence']
                identity = calculate_sequence_identity(seq1, seq2)
                blosum_score = calculate_blosum62_score(seq1, seq2)
                stability_diff = abs(sorted_data.iloc[j]['stability_score'] - sorted_data.iloc[i]['stability_score'])
                logger.info(f"Sample pair:")
                logger.info(f"- Sequence identity: {identity:.2f}")
                logger.info(f"- BLOSUM62 score: {blosum_score:.2f}")
                logger.info(f"- Stability difference: {stability_diff:.2f}")
        
        # Randomly sample pairs and randomly assign preference
        random.shuffle(valid_pairs)
        
        # Create balanced dataset with both preference directions
        n_pairs = len(valid_pairs) // 2  # We'll create balanced pairs
        self.paired_data = []
        
        # First half of pairs: teach model to prefer stable sequences
        # For these pairs, the more stable sequence (lower score) is marked as "chosen"
        # and gets the "high" stability tag
        for i in range(n_pairs):
            idx1, idx2 = valid_pairs[i]
            seq1, seq2 = sorted_data.iloc[idx1], sorted_data.iloc[idx2]
            if seq1['stability_score'] < seq2['stability_score']:  # Lower score = more stable
                chosen, rejected = seq1, seq2
            else:
                chosen, rejected = seq2, seq1
                
            self.paired_data.append({
                'chosen_sequence': chosen['sequence'],
                'chosen_perplexity': chosen['perplexity'], # we add perplexity for validation evals
                'rejected_sequence': rejected['sequence'],
                'rejected_perplexity': rejected['perplexity'], # we add perplexity for validation evals
                'chosen_score': chosen['stability_score'],
                'rejected_score': rejected['stability_score'],
                'ec_label': chosen['ec_label'],  # Both sequences have same EC number
                'prefer_stable': True
            })
        
        # Second half of pairs: teach model to prefer unstable sequences
        # For these pairs, the less stable sequence (higher score) is marked as "chosen"
        # and gets the "low" stability tag
        # for i in range(n_pairs, min(2 * n_pairs, len(valid_pairs))):
        #     idx1, idx2 = valid_pairs[i]
        #     seq1, seq2 = sorted_data.iloc[idx1], sorted_data.iloc[idx2]
        #     if seq1['stability_score'] > seq2['stability_score']:  # Higher score = less stable
        #         chosen, rejected = seq1, seq2
        #     else:
        #         chosen, rejected = seq2, seq1
                
        #     self.paired_data.append({
        #         'chosen_sequence': chosen['sequence'],
        #         'chosen_perplexity': chosen['perplexity'], # we add perplexity for validation evals
        #         'rejected_sequence': rejected['sequence'],
        #         'rejected_perplexity': rejected['perplexity'], # we add perplexity for validation evals
        #         'chosen_score': chosen['stability_score'],
        #         'rejected_score': rejected['stability_score'],
        #         'ec_label': chosen['ec_label'],
        #         'prefer_stable': False
        #     })
        
        # Convert to DataFrame for easier indexing
        self.paired_data = pd.DataFrame(self.paired_data)
        logger.info(f"Created {len(self.paired_data)} paired samples")
        logger.info(f"Number of prefer_stable=True: {sum(self.paired_data['prefer_stable'])}")
        
        # Log some statistics about the pairs
        score_diffs = abs(self.paired_data['chosen_score'] - self.paired_data['rejected_score'])
        logger.info(f"Average stability difference in pairs: {score_diffs.mean():.2f}")
        logger.info(f"Min stability difference in pairs: {score_diffs.min():.2f}")
        logger.info(f"Max stability difference in pairs: {score_diffs.max():.2f}")

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        pair = self.paired_data.iloc[idx]
        
        # Log every 1000th sample for monitoring
        if idx % 1000 == 0:
            logger.debug(f"DPO Sample {idx}:")
            logger.debug(f"Prefer stable: {pair['prefer_stable']}")
            logger.debug(f"Chosen score: {pair['chosen_score']:.2f}")
            logger.debug(f"Rejected score: {pair['rejected_score']:.2f}")
            logger.debug(f"Score difference: {abs(pair['chosen_score'] - pair['rejected_score']):.2f}")
        
        # Construct prompts with stability control tags
        stability_tag = None
        if self.type == "train":
            stability_tag = 'high' if pair['prefer_stable'] else 'low'

        chosen_prompt = self._format_sequence(pair['ec_label'], pair['chosen_sequence'], stability_tag)
        rejected_prompt = self._format_sequence(pair['ec_label'], pair['rejected_sequence'], stability_tag)
        
        # Tokenize both sequences
        chosen_inputs = self.tokenizer(
            chosen_prompt,
            max_length=self.max_length,
            # padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_inputs = self.tokenizer(
            rejected_prompt,
            max_length=self.max_length,
            # padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return only the necessary fields for DPO
        return {
            "chosen": {
                "input_ids": chosen_inputs["input_ids"].squeeze(0),
                "attention_mask": chosen_inputs["attention_mask"].squeeze(0),
                "stability_score": torch.tensor(pair['chosen_score'], dtype=torch.float),
                "perplexity": torch.tensor(pair['chosen_perplexity'], dtype=torch.float) # we add perplexity for validation evals
            },
            "rejected": {
                "input_ids": rejected_inputs["input_ids"].squeeze(0),
                "attention_mask": rejected_inputs["attention_mask"].squeeze(0),
                "stability_score": torch.tensor(pair['rejected_score'], dtype=torch.float),
                "perplexity": torch.tensor(pair['rejected_perplexity'], dtype=torch.float)
            }
        }
    