import csv
import torch
from tqdm import tqdm
import argparse
import os
import pandas as pd
import numpy as np

from datasets import load_dataset
from stability import stability_score_batch

def mutate_sequence_blosum(sequence, mutation_rate=0.1):
    """Apply BLOSUM62-based mutations to a sequence - mutates every position"""
    # BLOSUM62 matrix (simplified version for common amino acids)
    blosum62 = {
        'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
        'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
        'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
        'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
        'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
        'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
        'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
        'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
        'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
        'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
        'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
        'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
        'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
        'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
        'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
        'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
        'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
        'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
        'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
        'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4}
    }
    
    mutated_seq = list(sequence)
    
    # Go through each position and mutate it using BLOSUM probabilities
    for pos in range(len(sequence)):
        original_aa = sequence[pos]
        if original_aa in blosum62:
            # Get BLOSUM scores for this amino acid
            scores = blosum62[original_aa]
            # Convert negative scores to positive probabilities
            # Higher BLOSUM score = higher probability of mutation
            probs = []
            candidates = []
            for aa, score in scores.items():
                if aa != original_aa:  # Don't mutate to same amino acid
                    # Convert BLOSUM score to probability (higher score = higher prob)
                    prob = max(0.01, score + 5)  # Shift to make all positive
                    probs.append(prob)
                    candidates.append(aa)
            
            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            # Select mutation based on BLOSUM probabilities
            mutated_aa = np.random.choice(candidates, p=probs)
            mutated_seq[pos] = mutated_aa
    
    return ''.join(mutated_seq)

def load_brenda_sequences_from_csv(csv_path):
    """Load sequences from existing BRENDA CSV file"""
    df = pd.read_csv(csv_path)
    sequences = []
    for _, row in df.iterrows():
        sequences.append({
            'sequence': row['Sequence'],
            'ec_number': row['EC_Number'],
            'length': len(row['Sequence'])
        })
    return sequences

def load_brenda_sequences_from_hf(ec_number, min_length=180, max_length=260, limit=None):
    """Load sequences directly from HuggingFace dataset with same filtering as brenda.py"""
    print(f"Loading sequences for EC number: {ec_number}")
    print(f"Sequence length range: {min_length}-{max_length}")
    
    ds = load_dataset("AI4PD/ZymCTRL")
    filtered_ds = ds.filter(lambda x: x['text'].startswith(ec_number))

    def extract_sequence(text):
        try:
            ec_number, sequence = text.split('<sep>')[:2]
            sequence = sequence.split('<start>')[1].split('<end>')[0]
            return sequence
        except (IndexError, ValueError) as e:
            print(f"Error extracting sequence from text: {e}")
            return None
    
    filtered_ds = filtered_ds.map(lambda x: {'text': extract_sequence(x['text'])})
    
    # Remove entries where text is None
    filtered_ds = filtered_ds.filter(lambda x: x['text'] is not None)
    
    # Apply length filters (same as brenda.py)
    filtered_ds = filtered_ds.filter(lambda x: len(x['text']) <= max_length)
    filtered_ds = filtered_ds.filter(lambda x: len(x['text']) >= min_length)
    filtered_ds = filtered_ds.map(lambda x: {'length': len(x['text'])})
    filtered_ds = filtered_ds.sort('length', reverse=False)
    
    print(f"Found {len(filtered_ds['train'])} sequences after filtering")
    
    # Convert to list format
    sequences = []
    total_items = len(filtered_ds['train']) if limit is None else min(limit, len(filtered_ds['train']))
    
    for i, item in enumerate(filtered_ds['train']):
        if limit is not None and i >= limit:
            break
        sequences.append({
            'sequence': item['text'],
            'ec_number': ec_number,
            'length': item['length']
        })
    
    return sequences

def count_sequence_differences(seq1, seq2):
    """Count the number of different amino acids between two sequences"""
    if len(seq1) != len(seq2):
        return None  # Can't compare sequences of different lengths
    return sum(1 for a, b in zip(seq1, seq2) if a != b)

def generate_mutations_for_brenda(sequences, n_mutations_per_seq=5, mutation_rates=None, score_mutations=True, scoring_method="rosetta", batch_size=8):
    """Generate BLOSUM mutations for each BRENDA sequence using batched processing"""
    if mutation_rates is None:
        mutation_rates = [1]  # Different mutation rates
    
    all_data = []
    
    print(f"Generating {n_mutations_per_seq} mutations per sequence using {len(mutation_rates)} different mutation rates")
    print(f"Processing in batches of {batch_size} sequences")
    
    # Process sequences in batches
    for batch_start in tqdm(range(0, len(sequences), batch_size), desc="Processing sequence batches"):
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]
        
        # Collect all original sequences in this batch
        original_seqs = [seq_info['sequence'] for seq_info in batch_sequences]
        
        # Score all original sequences in one batch
        original_scores_batch = None
        if score_mutations:
            try:
                print(f"Scoring {len(original_seqs)} original sequences...")
                stability_results = stability_score_batch(original_seqs, scoring_method=scoring_method)
                original_scores_batch = []
                
                for (raw_score, dg, plddt) in stability_results:
                    # Assign stability label for original
                    if scoring_method == "rosetta":
                        if dg < -50:
                            stability_label = "high"
                        elif dg > 0:
                            stability_label = "low"
                        else:
                            stability_label = "medium"
                    else:  # esm-if
                        if dg < -2.0:
                            stability_label = "high"
                        elif dg > 0.0:
                            stability_label = "low"
                        else:
                            stability_label = "medium"
                    
                    original_scores_batch.append({
                        'Original_Raw_Score': raw_score,
                        'Original_DeltaG': dg,
                        'Original_Stability_Label': stability_label,
                        'Original_pLDDT': plddt,
                        'Original_Scoring_Method': scoring_method
                    })
            except Exception as e:
                print(f"Error scoring original sequences in batch: {e}")
                original_scores_batch = [None] * len(original_seqs)
        
        # Generate mutations for all sequences in batch
        all_mutations_batch = []
        mutation_tracking = []  # Store (original_seq_idx, mutation_num, original_seq, ec_number)
        
        for seq_idx, seq_info in enumerate(batch_sequences):
            original_seq = seq_info['sequence']
            ec_number = seq_info['ec_number']
            
            # Generate mutations for this sequence
            for i in range(n_mutations_per_seq):
                mutation_rate = mutation_rates[i % len(mutation_rates)]
                mutated_seq = mutate_sequence_blosum(original_seq, mutation_rate)
                
                all_mutations_batch.append(mutated_seq)
                mutation_tracking.append((seq_idx, i, original_seq, ec_number))
        
        # Score all mutations in one large batch
        mutated_scores_batch = None
        if score_mutations and all_mutations_batch:
            try:
                print(f"Scoring {len(all_mutations_batch)} mutated sequences...")
                stability_results = stability_score_batch(all_mutations_batch, scoring_method=scoring_method)
                mutated_scores_batch = []
                
                for (raw_score, dg, plddt) in stability_results:
                    # Assign stability label for mutated sequence
                    if scoring_method == "rosetta":
                        if dg < -50:
                            stability_label = "high"
                        elif dg > 0:
                            stability_label = "low"
                        else:
                            stability_label = "medium"
                    else:  # esm-if
                        if dg < -2.0:
                            stability_label = "high"
                        elif dg > 0.0:
                            stability_label = "low"
                        else:
                            stability_label = "medium"
                    
                    mutated_scores_batch.append({
                        'Mutated_Raw_Score': raw_score,
                        'Mutated_DeltaG': dg,
                        'Mutated_Stability_Label': stability_label,
                        'Mutated_pLDDT': plddt,
                        'Mutated_Scoring_Method': scoring_method
                    })
            except Exception as e:
                print(f"Error scoring mutated sequences in batch: {e}")
                mutated_scores_batch = [None] * len(all_mutations_batch)
        
        # Assemble results using the tracking information
        for mutation_idx, (seq_idx, mut_num, original_seq, ec_number) in enumerate(mutation_tracking):
            original_scores = original_scores_batch[seq_idx] if original_scores_batch else None
            mutated_scores = mutated_scores_batch[mutation_idx] if mutated_scores_batch else None
            mutated_seq = all_mutations_batch[mutation_idx]
            
            # Count differences
            differences_count = count_sequence_differences(original_seq, mutated_seq)
            
            # Create row data
            row_data = {
                'EC_Number': ec_number,
                'Original_Sequence': original_seq,
                'Mutated_Sequence': mutated_seq,
                'Differences_Count': differences_count,
                'Original_pLDDT': original_scores['Original_pLDDT'] if original_scores else None,
                'Mutated_pLDDT': mutated_scores['Mutated_pLDDT'] if mutated_scores else None,
                'Original_Rosetta_Score': original_scores['Original_DeltaG'] if original_scores else None,
                'Mutated_Rosetta_Score': mutated_scores['Mutated_DeltaG'] if mutated_scores else None
            }
            
            all_data.append(row_data)
        
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_data)

def save_results(df, output_path):
    """Save results to CSV and generate summary statistics"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save main CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Generate summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"Total sequence pairs processed: {len(df)}")
    print(f"Unique original sequences: {df['Original_Sequence'].nunique()}")
    
    if 'Differences_Count' in df.columns and df['Differences_Count'].notna().any():
        print(f"\nSequence differences:")
        print(f"Average differences per pair: {df['Differences_Count'].mean():.1f}")
        print(f"Min differences: {df['Differences_Count'].min()}")
        print(f"Max differences: {df['Differences_Count'].max()}")
        print(f"Differences distribution:")
        print(df['Differences_Count'].value_counts().sort_index().head(10))
    
    if 'Original_Rosetta_Score' in df.columns and df['Original_Rosetta_Score'].notna().any():
        print("\nRosetta Score statistics:")
        print("Original sequences:")
        print(df['Original_Rosetta_Score'].describe())
        print("\nMutated sequences:")
        print(df['Mutated_Rosetta_Score'].describe())
        
        # Calculate score differences
        score_diff = df['Mutated_Rosetta_Score'] - df['Original_Rosetta_Score']
        improved_count = (score_diff < 0).sum()
        degraded_count = (score_diff > 0).sum()
        unchanged_count = (score_diff == 0).sum()
        
        print(f"\nStability changes:")
        print(f"Improved: {improved_count}")
        print(f"Degraded: {degraded_count}")
        print(f"Unchanged: {unchanged_count}")
        
        if improved_count > 0:
            best_improvement = score_diff.min()
            print(f"Best improvement: {best_improvement:.2f}")
        
        if degraded_count > 0:
            worst_degradation = score_diff.max()
            print(f"Worst degradation: {worst_degradation:.2f}")
    
    if 'Original_pLDDT' in df.columns and df['Original_pLDDT'].notna().any():
        print("\npLDDT statistics:")
        print("Original sequences:")
        print(df['Original_pLDDT'].describe())
        print("\nMutated sequences:")
        print(df['Mutated_pLDDT'].describe())
    
    # Save FASTA files
    base_path = output_path.replace('.csv', '')
    
    # Original sequences FASTA
    original_sequences = df[['EC_Number', 'Original_Sequence']].drop_duplicates()
    with open(f"{base_path}_original.fasta", 'w') as f:
        for idx, row in original_sequences.iterrows():
            f.write(f">{row['EC_Number']}_original_{idx}\n")
            f.write(f"{row['Original_Sequence']}\n")
    
    # Mutated sequences FASTA
    with open(f"{base_path}_mutated.fasta", 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{row['EC_Number']}_mutated_{idx}\n")
            f.write(f"{row['Mutated_Sequence']}\n")
    
    print(f"\nFASTA files saved:")
    print(f"- Original sequences: {base_path}_original.fasta")
    print(f"- Mutated sequences: {base_path}_mutated.fasta")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BLOSUM-mutated sequences from BRENDA dataset")
    parser.add_argument("--ec_number", type=str, default="4.2.1.1", help="EC number to filter sequences")
    parser.add_argument("--min_length", type=int, default=180, help="Minimum length of sequences to process")
    parser.add_argument("--max_length", type=int, default=260, help="Maximum length of sequences to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sequences to process")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to existing BRENDA CSV file (if not provided, will load from HuggingFace)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--n_mutations", type=int, default=5, help="Number of mutations to generate per sequence")
    parser.add_argument("--mutation_rates", type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.25], 
                      help="List of mutation rates to use")
    parser.add_argument("--no_scoring", action="store_true", help="Skip stability scoring (faster)")
    parser.add_argument("--scoring_method", type=str, default="rosetta", choices=["esm-if", "rosetta"],
                      help="Scoring method for stability evaluation: 'rosetta' (default) or 'esm-if'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing sequences (affects memory usage)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Starting BRENDA sequence mutation pipeline...")
    print(f"Parameters:")
    print(f"- EC number: {args.ec_number}")
    print(f"- Length range: {args.min_length}-{args.max_length}")
    print(f"- Mutations per sequence: {args.n_mutations}")
    print(f"- Mutation rates: {args.mutation_rates}")
    print(f"- Scoring: {'Disabled' if args.no_scoring else args.scoring_method.upper()}")
    print(f"- Random seed: {args.seed}")
    print(f"- Batch size: {args.batch_size}")
    
    # Load BRENDA sequences
    if args.input_csv:
        print(f"Loading sequences from CSV: {args.input_csv}")
        sequences = load_brenda_sequences_from_csv(args.input_csv)
    else:
        print("Loading sequences from HuggingFace dataset...")
        sequences = load_brenda_sequences_from_hf(
            args.ec_number, 
            min_length=args.min_length, 
            max_length=args.max_length, 
            limit=args.limit
        )
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Generate mutations
    df = generate_mutations_for_brenda(
        sequences, 
        n_mutations_per_seq=args.n_mutations,
        mutation_rates=args.mutation_rates,
        score_mutations=not args.no_scoring,
        scoring_method=args.scoring_method,
        batch_size=args.batch_size
    )
    
    # Save results
    save_results(df, args.output_path)
    
    print("\nPipeline completed successfully!") 