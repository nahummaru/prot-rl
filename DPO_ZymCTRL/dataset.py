import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import logging
from typing import Optional, List
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
        type: str = "train",
        use_control_tags: bool = True,
        include_stability_levels: Optional[List[str]] = None
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
        self.use_control_tags = use_control_tags
        self.include_stability_levels = include_stability_levels or ['high', 'medium', 'low']
        logger.info(f"Using stability threshold: {stability_threshold}")
        
        # Load data
        if isinstance(data_path, list):
            # Load and concatenate multiple CSV files
            dfs = []
            for path in data_path:
                df = pd.read_csv(path)
                dfs.append(df)
            self.data = pd.concat(dfs, ignore_index=True)
        else:
            # Load single CSV file
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
        if self.use_control_tags and stability_level in self.include_stability_levels:
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
        split_percent=.15,
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
            if 'high' in self.include_stability_levels:
                self.sample_indices.append((idx, 'high'))  # Most stable = high stability
        for idx in middle_indices:
            if 'medium' in self.include_stability_levels:
                self.sample_indices.append((idx, 'medium'))
        for idx in bottom_indices:
            if 'low' in self.include_stability_levels:
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
        
        prompt = self._format_sequence(sample['ec_label'], sample['sequence'], stability_level)

        # FOR INTERNAL DEBUGGING
        print(prompt)
        print(f"Stability score: {sample['stability_score']:.2f}")
        
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
            "attention_mask": inputs["attention_mask"].squeeze(0)
        } 

# DPO layer
class ZymCTRLDPODataset(ZymCTRLDataset):
    def __init__(
        self,
        min_sequence_identity: float = 0.05,  # Minimum sequence identity (90%)
        min_blosum62_score: float = -1.0,    # Minimum BLOSUM62 score per residue
        split_percent: float = 0.25,
        n_pairs_to_sample: int = 50,  # Number of pairs to sample and create
        max_sampling_attempts: int = 10000,  # Maximum number of attempts to find valid pairs
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_sequence_identity = min_sequence_identity
        self.min_blosum62_score = min_blosum62_score

        logger.info(f"Initializing DPO ZymCTRLDataset.")
        logger.info(f"Using sequence identity threshold: {min_sequence_identity}")
        logger.info(f"Using BLOSUM62 score threshold: {min_blosum62_score}")
        logger.info(f"Will attempt to create {n_pairs_to_sample} valid pairs")
        
        # Sort data by stability score for easier pairing
        sorted_data = self.data.sort_values('stability_score', ascending=True)
        n_samples = len(sorted_data)
        
        # Select top and bottom sequences
        n_subset = int(n_samples * split_percent)
        top_indices = list(range(n_subset))  # Most stable (lowest deltaG)
        bottom_indices = list(range(n_samples - n_subset, n_samples))  # Least stable (highest deltaG)
        
        # Calculate maximum possible pairs
        max_possible_pairs = min(len(top_indices), len(bottom_indices))
        if n_pairs_to_sample > max_possible_pairs:
            logger.warning(f"Requested {n_pairs_to_sample} pairs but only {max_possible_pairs} possible "
                         f"when each sequence can only be used once. Reducing target to {max_possible_pairs}.")
            n_pairs_to_sample = max_possible_pairs
        
        logger.info(f"Using top and bottom {n_subset} samples each ({split_percent*100}% of total)")
        logger.info(f"Stability ranges:")
        logger.info(f"Top (most stable): {sorted_data.iloc[top_indices]['stability_score'].min():.2f} to {sorted_data.iloc[top_indices]['stability_score'].max():.2f}")
        logger.info(f"Bottom (least stable): {sorted_data.iloc[bottom_indices]['stability_score'].min():.2f} to {sorted_data.iloc[bottom_indices]['stability_score'].max():.2f}")
        
        # Pre-compute all stability differences between top and bottom groups
        logger.info("Pre-computing stability differences...")
        top_scores = sorted_data.iloc[top_indices]['stability_score'].values
        bottom_scores = sorted_data.iloc[bottom_indices]['stability_score'].values
        
        # Create meshgrid for vectorized computation
        top_mesh, bottom_mesh = np.meshgrid(top_scores, bottom_scores)
        stability_diffs = np.abs(bottom_mesh - top_mesh)
        
        # Get valid stability difference pairs
        valid_stability_pairs = np.argwhere(stability_diffs >= self.stability_threshold)
        
        if len(valid_stability_pairs) == 0:
            raise ValueError(
                f"No pairs found meeting stability threshold >= {self.stability_threshold}\n"
                f"Max stability difference: {stability_diffs.max():.2f}"
            )
            
        logger.info(f"Found {len(valid_stability_pairs)} pairs meeting stability threshold")
        
        # Shuffle the valid pairs
        np.random.shuffle(valid_stability_pairs)
        
        # Now process these pairs in order until we find enough valid ones
        valid_pairs = []
        used_top_indices = set()
        used_bottom_indices = set()
        processed_count = 0
        
        logger.info("Finding valid pairs meeting all criteria...")
        for bottom_idx_local, top_idx_local in valid_stability_pairs:
            # Convert local indices to global indices
            top_idx = top_indices[top_idx_local]
            bottom_idx = bottom_indices[bottom_idx_local]
            
            # Skip if either sequence is already used
            if top_idx in used_top_indices or bottom_idx in used_bottom_indices:
                continue
                
            # Get sequences
            seq1 = sorted_data.iloc[top_idx]['sequence']
            seq2 = sorted_data.iloc[bottom_idx]['sequence']
            
            # Check sequence identity
            identity = calculate_sequence_identity(seq1, seq2)
            if identity < self.min_sequence_identity:
                continue
                
            # Check BLOSUM62 score
            blosum_score = calculate_blosum62_score(seq1, seq2)
            if blosum_score < self.min_blosum62_score:
                continue
            
            # Valid pair found
            valid_pairs.append((top_idx, bottom_idx))
            used_top_indices.add(top_idx)
            used_bottom_indices.add(bottom_idx)
            
            processed_count += 1
            
            # Log progress every 1000 pairs
            if processed_count % 1000 == 0:
                logger.info(f"Processed {processed_count} pairs, found {len(valid_pairs)} valid ones")
            
            # Check if we have enough pairs
            if len(valid_pairs) >= n_pairs_to_sample:
                break
        
        if len(valid_pairs) == 0:
            raise ValueError(
                f"No valid pairs found meeting all criteria:\n"
                f"- Stability difference >= {self.stability_threshold}\n"
                f"- Sequence identity >= {self.min_sequence_identity}\n"
                f"- BLOSUM62 score >= {self.min_blosum62_score}"
            )
        
        if len(valid_pairs) < n_pairs_to_sample:
            logger.warning(f"Could only find {len(valid_pairs)} valid pairs, "
                         f"which is less than the requested {n_pairs_to_sample} pairs")
        
        logger.info(f"Found {len(valid_pairs)} valid pairs meeting all criteria")
        logger.info(f"Used {len(used_top_indices)} unique sequences from top group")
        logger.info(f"Used {len(used_bottom_indices)} unique sequences from bottom group")
        
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
        
        # Create balanced dataset with both preference directions
        self.paired_data = []
        
        # Create pairs: teach model to prefer stable sequences
        # For these pairs, the more stable sequence (lower score) is marked as "chosen"
        # and gets the "high" stability tag
        for i, j in valid_pairs:
            seq1, seq2 = sorted_data.iloc[i], sorted_data.iloc[j]
            if seq1['stability_score'] < seq2['stability_score']:  # Lower score = more stable
                chosen, rejected = seq1, seq2
            else:
                chosen, rejected = seq2, seq1
                
            self.paired_data.append({
                'chosen_sequence': chosen['sequence'],
                'chosen_perplexity': chosen.get('perplexity', 0), # default to 0 if perplexity not in data
                'rejected_sequence': rejected['sequence'], 
                'rejected_perplexity': rejected.get('perplexity', 0), # default to 0 if perplexity not in data
                'chosen_score': chosen['stability_score'],
                'rejected_score': rejected['stability_score'],
                'ec_label': chosen['ec_label'],  # Both sequences have same EC number
                'prefer_stable': True
            })
        
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
        if self.use_control_tags:
            stability_tag = 'high' if pair['prefer_stable'] else 'low'
        chosen_prompt = self._format_sequence(pair['ec_label'], pair['chosen_sequence'], stability_tag)
        rejected_prompt = self._format_sequence(pair['ec_label'], pair['rejected_sequence'], stability_tag)

        if idx == 0:
            print("chosen prompt: ", chosen_prompt)
            print("rejected prompt: ", rejected_prompt)
            print("-------")
        
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