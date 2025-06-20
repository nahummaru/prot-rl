import torch, os, math, argparse
from transformers import GPT2LMHeadModel, AutoTokenizer
from stability import stability_score, stability_score_batch
import pandas as pd
from tqdm import tqdm
import json
from typing import Optional

from utils import perplexity_from_logits

# Configure PyTorch memory management to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

AVOID_ESM = False

def remove_characters(sequence, char_list):
    '''
    Removes special tokens used during training.
    '''
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids, model):
    '''
    Computes perplexities for the generated sequences. 
    '''
    with torch.no_grad():
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def format_sequence(ec_label: str, sequence: str, stability_level: Optional[str] = None) -> str:
    """Format sequence according to original ZymCTRL paper format"""
    if stability_level is not None:
        # Add stability control tag right after EC label
        return f"{ec_label}<stability={stability_level}><sep><start>"
    return f"{ec_label}<sep><start>>"

def generate_pivot_sequences(label, model, special_tokens, device, tokenizer, plddt_threshold, perplexity_threshold, min_length, max_length, n_continuations=8, control_tag=""):
    '''
    Generate sequences using the pivot-based approach:
    1. Generate a base sequence
    2. Select a pivot point
    3. Generate multiple continuations from that pivot
    4. Evaluate stability differences
    '''
    print(f"Generating pivot-based sequences for label: {label}")
    
    # Extract stability level from control tag if present
    stability_level = None
    if control_tag:
        if control_tag.startswith('<stability=') and control_tag.endswith('>'):
            stability_level = control_tag[11:-1]  # Extract 'high', 'medium', or 'low'
    
    # Format input with proper structure
    input_text = format_sequence(label, "", stability_level)  # Empty sequence for generation

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    base_output = model.generate(
        input_ids,
        top_k=9,
        repetition_penalty=1.2,
        max_length=max_length,
        min_length=min_length,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=n_continuations,
        temperature=1,
        no_repeat_ngram_size=3
    )
    
    # base_sequence = remove_characters(tokenizer.decode(base_output), special_tokens)
    
    # Select a pivot point
    # Find the minimum sequence length by looking at non-padding tokens
    # (assuming padding token is 0)
    non_pad_mask = (base_output != 0).int()
    seq_lengths = non_pad_mask.sum(dim=1)
    min_seq_length = seq_lengths.min().item()
    
    # Generate pivot point that's shorter than the minimum sequence length
    # Use 25-75% of the minimum length to ensure good context
    pivot = torch.randint(int(min_seq_length * 0.25), int(min_seq_length * 0.75), (1,)).item()
    
    # Get the prefix up to the pivot point
    prefix_ids = base_output[:,:pivot]
    
    continuations = model.generate(
        prefix_ids,
        top_k=9,
        repetition_penalty=1.2,
        max_length=max_length,
        min_length=min_length,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=1,
        temperature=1.2,  # Slightly higher temperature for more diversity
        no_repeat_ngram_size=3
    )
    
    # Process continuations
    sequences = []
    for output in continuations:
        decoded_output = tokenizer.decode(output)
        sequence = remove_characters(decoded_output, special_tokens)
        
        # Calculate perplexity
        logits = model.forward(output).logits
        ppl = perplexity_from_logits(logits, output, None).item()
        
        sequences.append((sequence, ppl))
    # Print each sequence and its perplexity
    print("\nGenerated sequences and their perplexities:")
    for i, (seq, ppl) in enumerate(sequences, 1):
        print(f"\nSequence {i}:")
        print(f"Perplexity: {ppl:.2f}")
        print(f"Sequence: {seq}")
    print("--------------------------------")
    
    # Sort by perplexity
    sequences.sort(key=lambda x: x[1])
    
    return sequences

def generate_blosum_sequences(label, model, special_tokens, device, tokenizer, plddt_threshold, perplexity_threshold, min_length, max_length, n_continuations=8, control_tag=""):
    '''
    Generate sequences using the BLOSUM-based approach:
    1. Generate a base sequence
    2. Apply BLOSUM matrix-based mutations
    3. Generate multiple continuations from mutated sequences
    4. Evaluate stability differences
    '''
    import numpy as np
    
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
    
    def mutate_sequence_blosum(sequence, mutation_rate=0.1):
        """Apply BLOSUM62-based mutations to a sequence"""
        mutated_seq = list(sequence)
        n_mutations = max(1, int(len(sequence) * mutation_rate))
        
        # Select random positions to mutate
        positions = np.random.choice(len(sequence), n_mutations, replace=False)
        
        for pos in positions:
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
    
    print(f"Generating BLOSUM-based sequences for label: {label}")
    
    # Extract stability level from control tag if present
    stability_level = None
    if control_tag:
        if control_tag.startswith('<stability=') and control_tag.endswith('>'):
            stability_level = control_tag[11:-1]  # Extract 'high', 'medium', or 'low'
    
    # Format input with proper structure
    input_text = format_sequence(label, "", stability_level)  # Empty sequence for generation

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    base_output = model.generate(
        input_ids,
        top_k=9,
        repetition_penalty=1.2,
        max_length=max_length,
        min_length=min_length,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=1,  # Generate one base sequence
        temperature=1,
        no_repeat_ngram_size=3
    )
    
    # Get base sequence
    base_decoded = tokenizer.decode(base_output[0])
    base_sequence = remove_characters(base_decoded, special_tokens)
    
    # Generate multiple BLOSUM-mutated variants
    sequences = []
    for i in range(n_continuations):
        # Apply BLOSUM-based mutations
        mutated_sequence = mutate_sequence_blosum(base_sequence, mutation_rate=0.05 + i * 0.02)  # Varying mutation rates
        
        # Re-encode the mutated sequence for continuation generation
        mutated_input = format_sequence(label, mutated_sequence[:len(mutated_sequence)//2], stability_level)
        mutated_ids = tokenizer.encode(mutated_input, return_tensors='pt').to(device)
        
        # Generate continuation from mutated sequence
        continuation = model.generate(
            mutated_ids,
            top_k=9,
            repetition_penalty=1.2,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=1,
            pad_token_id=0,
            do_sample=True,
            num_return_sequences=1,
            temperature=1.2,  # Slightly higher temperature for more diversity
            no_repeat_ngram_size=3
        )
        
        # Process continuation
        decoded_output = tokenizer.decode(continuation[0])
        final_sequence = remove_characters(decoded_output, special_tokens)
        
        # Calculate perplexity
        logits = model.forward(continuation[0]).logits
        ppl = perplexity_from_logits(logits, continuation[0], None).item()
        
        sequences.append((final_sequence, ppl))
    
    # Print each sequence and its perplexity
    print("\nGenerated BLOSUM-mutated sequences and their perplexities:")
    for i, (seq, ppl) in enumerate(sequences, 1):
        print(f"\nSequence {i}:")
        print(f"Perplexity: {ppl:.2f}")
        print(f"Sequence: {seq}")
    print("--------------------------------")
    
    # Sort by perplexity
    sequences.sort(key=lambda x: x[1])
    
    return sequences

def main(label, model, special_tokens, device, tokenizer, plddt_threshold, perplexity_threshold, min_length, max_length, use_pivot=False, n_continuations=5, control_tag=""):
    '''
    Function to generate sequences from the loaded model.
    '''
    if use_pivot:
        sequences = generate_pivot_sequences(label, model, special_tokens, device, tokenizer,
                                          plddt_threshold, perplexity_threshold, min_length, max_length, n_continuations, control_tag)
    else:
        print(f"Generating sequences for label: {label}")
        # Extract stability level from control tag if present
        stability_level = None
        if control_tag:
            if control_tag.startswith('<stability=') and control_tag.endswith('>'):
                stability_level = control_tag[11:-1]  # Extract 'high', 'medium', or 'low'
        
        # Format input with proper structure
        input_text = format_sequence(label, "", stability_level)  # Empty sequence for generation
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        # Generating sequences
        outputs = model.generate(
            input_ids, 
            top_k=9, 
            repetition_penalty=1.2,
            max_length=1024,
            min_length=10,
            eos_token_id=1,
            pad_token_id=0,
            do_sample=True,
            num_return_sequences=20,  
            temperature=1,
            no_repeat_ngram_size=3
        ) 
        print(f"Generated {len(outputs)} raw sequences")
        
        # Check sequence sanity, ensure sequences are properly terminated
        new_outputs = [output for output in outputs if output[-1] == 0 or output[-1] == 1]  # Accept either PAD or EOS token
        
        if not new_outputs:
            print("Warning: No properly terminated sequences found!")
            return {}

        print(f"After filtering: {len(new_outputs)} valid sequences")

        ppls = []
        for output in new_outputs:
            decoded_output = tokenizer.decode(output)

            logits = model.forward(output).logits
            ppl = perplexity_from_logits(logits, output, None).item()

            ppls.append((decoded_output, ppl))

        # Sort the batch by perplexity, the lower the better
        ppls.sort(key=lambda i:i[1]) 

        sequences = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    # Final results dictionary
    result = {}
    result[label] = sequences
    return result

def process_sequences_with_stability(sequences_dict, plddt_threshold, perplexity_threshold, min_length, max_length, disable_filtering=False, scoring_method="rosetta"):
    """
    Add stability scores to sequences and convert to DataFrame.
    Processes sequences in batches using stability_score_batch.
    """
    data = []
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Process sequences in batches
    total_sequences = sum(len(seq_list) for seq_list in sequences_dict.values())
    processed = 0
    filtered_out = 0

    batch_size = 8  # Adjust based on available GPU memory
    current_batch = []
    current_batch_info = []  # Store (label, seq, ppl) for each sequence in batch
    
    print(f"Using {scoring_method.upper()} scoring method for stability evaluation")
    
    for label, seq_list in tqdm(sequences_dict.items(), desc="Scoring sequences"):
        for seq, ppl in seq_list:
            processed += 1
            
            # Check if sequence is valid
            if not disable_filtering and not all(c in canonical_amino_acids for c in seq):
                print(f"Skipping invalid sequence: {seq[:20]}...")
                filtered_out += 1
                continue
            
            # Apply length filter
            if not disable_filtering and (len(seq) < min_length or len(seq) > max_length):
                print(f"Filtering out sequence due to length: {len(seq)}")
                filtered_out += 1
                continue

            # Apply perplexity filter
            if not disable_filtering and ppl >= perplexity_threshold:
                print(f"Filtering out sequence due to high perplexity: {ppl}")
                filtered_out += 1
                continue
            
            print(f"\nProcessing sequence {processed}/{total_sequences}")
            
            try:
                # Add sequence to current batch
                current_batch.append(seq)
                current_batch_info.append((label, seq, ppl))
                
                # Process batch when it's full or on the last sequence
                if len(current_batch) == batch_size or processed == total_sequences:
                    # Get stability scores for the batch
                    if AVOID_ESM:
                        stability_results = [(torch.rand(1).item() * 2 - 1, 
                                           torch.rand(1).item() * 2 - 1,
                                           torch.rand(1).item()) for _ in range(len(current_batch))]
                    else:
                        stability_results = stability_score_batch(current_batch, scoring_method=scoring_method)
                    
                    # Process results for each sequence in the batch
                    for (label, seq, ppl), (raw_score, dg, plddt) in zip(current_batch_info, stability_results):
                        # Apply pLDDT filter
                        if not disable_filtering and plddt < plddt_threshold:
                            print(f"Filtering out sequence due to low pLDDT: {plddt}")
                            filtered_out += 1
                            continue

                        # Use deltaG as the stability score
                        stability = dg
                        
                        # Assign stability label based on deltaG
                        # Note: For Rosetta, more negative energy = more stable
                        # For ESM-IF, the fit parameters already convert to kcal/mol with proper sign
                        if scoring_method == "rosetta":
                            # Rosetta energy units: more negative = more stable
                            if stability < -50:  # Very stable
                                stability_label = "high"
                            elif stability > 0:   # Unstable
                                stability_label = "low"
                            else:                 # Moderately stable
                                stability_label = "medium"
                        else:  # esm-if
                            # ESM-IF converted deltaG: more negative = more stable
                            if stability < -2.0:  # More negative = more stable
                                stability_label = "high"
                            elif stability > 0.0:
                                stability_label = "low"
                            else:
                                stability_label = "medium"

                        print(f"seq: {seq}")
                        print(f"Scoring method: {scoring_method}, Raw score: {raw_score:.2f}, ΔG: {dg:.2f}, Label: {stability_label}")
                            
                        data.append({
                            'sequence': seq,
                            'perplexity': ppl,
                            'stability_score': stability,
                            'stability_raw_score': raw_score,
                            'stability_label': stability_label,
                            'ec_label': label,
                            'plddt': plddt,
                            'scoring_method': scoring_method
                        })
                    
                    # Clear batch
                    current_batch = []
                    current_batch_info = []
                    
                    # Clear GPU memory after each batch
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing sequence: {str(e)}")
                print("Skipping problematic sequence and continuing...")
                filtered_out += 1
                continue
    
    print(f"\nQuality control: Filtered out {filtered_out}/{total_sequences} sequences ({filtered_out/total_sequences:.2%})")
    return pd.DataFrame(data)

def save_data(df, output_dir="training_data"):
    """
    Save processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full dataset
    print("--------------------------------")
    print(f"LOG: Saving full dataset to csv at {output_dir}/sequences.csv")
    print("--------------------------------")

    df.to_csv(f"{output_dir}/sequences_{args.control_tag.replace('<', '').replace('>', '')}.csv", index=False)
    
    # Save FASTA files by stability label
    for label in ['high', 'low', 'medium']:
        subset = df[df['stability_label'] == label]
        with open(f"{output_dir}/stability_{label}.fasta", 'w') as f:
            for idx, row in subset.iterrows():
                scoring_method = row.get('scoring_method', 'unknown')
                f.write(f">{row['ec_label']}_{idx}_stability={label}_score={row['stability_score']:.2f}_method={scoring_method}\n")
                f.write(f"{row['sequence']}\n")
    
    print(f"\nDataset statistics:")
    print(f"Total sequences: {len(df)}")
    print("\nStability distribution:")
    print(df['stability_label'].value_counts())
    print("\nStability score statistics:")
    print(df['stability_score'].describe())

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, default=0)
    parser.add_argument("--ec_label", type=str, default="4.2.1.1")
    parser.add_argument("--n_batches", type=int, default=1)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model_path", type=str, default="",
                      help="Optional: Path to specific model checkpoint to use. If not provided, uses iteration-based loading.")
    parser.add_argument("--checkpoint_path", type=str, default="",
                      help="Optional: Path to a specific model checkpoint to load.")
    parser.add_argument("--data_type", type=str, default="train", choices=["train", "val"],
                      help="Specify whether this is training or validation data. Affects output directory prefix.")
    parser.add_argument("--plddt_threshold", type=float, default=0.8,
                      help="Minimum pLDDT score for sequence filtering")
    parser.add_argument("--perplexity_threshold", type=float, default=5,
                      help="Maximum perplexity score for sequence filtering")
    parser.add_argument("--min_length", type=int, default=100,
                      help="Minimum sequence length for filtering")
    parser.add_argument("--max_length", type=int, default=300,
                      help="Maximum sequence length for filtering")
    parser.add_argument("--use_pivot", action="store_true",
                      help="Use pivot-based sequence generation")
    parser.add_argument("--n_continuations", type=int, default=5,
                      help="Number of continuations to generate when using pivot-based generation")
    parser.add_argument("--disable_filtering", action="store_true", default=True,
                      help="Disable sequence filtering based on pLDDT and perplexity thresholds")
    parser.add_argument("--control_tag", type=str, default="",
                      help="Control tag to use for all generations (e.g. '<stability=high>')")
    parser.add_argument("--sequence_path", type=str, default="", 
                      help="Path to pre-generated sequences")
    parser.add_argument("--scoring_method", type=str, default="rosetta", choices=["esm-if", "rosetta"],
                      help="Scoring method for stability evaluation: 'rosetta' (default) or 'esm-if'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {args.scoring_method.upper()} scoring method")

    if args.sequence_path != "":
        with open(args.sequence_path, "r") as f:
            all_sequences = json.load(f)
    else:
        # Load model and tokenizer
        if args.model_path:
            print(f"Loading model from checkpoint: {args.model_path}")
            if args.model_path.endswith('.ckpt'):
                # First load base model architecture
                model = GPT2LMHeadModel.from_pretrained('AI4PD/ZymCTRL').to(device)
                tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL')

                special_tokens_dict = {
                    "additional_special_tokens": [
                        "<stability=high>",
                        "<stability=medium>",
                        "<stability=low>"
                    ]
                }
                tokenizer.add_special_tokens(special_tokens_dict)
                model.resize_token_embeddings(len(tokenizer))
        
                
                # Load checkpoint state dict
                checkpoint = torch.load(args.model_path, map_location=device)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                    
                # Remove 'model.' prefix from state dict keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_key = k[6:]  # Remove 'model.' prefix
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                        
                # Load the processed state dict
                model.load_state_dict(new_state_dict)
                print("Successfully loaded checkpoint weights")
            else:
                model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        else:
            # Use iteration-based or default model
            if args.iteration_num == 0:
                model_name = 'AI4PD/ZymCTRL'
            else:
                model_name = f'output_iteration{args.iteration_num}'
            
            print(f"Loading model: {model_name}")
            model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Only add special tokens for base model
            if model_name == 'AI4PD/ZymCTRL':
                special_tokens_dict = {
                    "additional_special_tokens": [
                        "<stability=high>",
                        "<stability=medium>",
                        "<stability=low>"
                    ]
                }
                tokenizer.add_special_tokens(special_tokens_dict)
                model.resize_token_embeddings(len(tokenizer))

        special_tokens = ['<start>', '<end>', '<|endoftext|>', '<pad>', '<sep>', ' ']

        # Generate sequences in batches
        all_sequences = {}
        all_sequences[args.ec_label] = []
        
        print(f"Generating {args.n_batches} batch(es) of sequences")
        for i in tqdm(range(args.n_batches), desc="Generating sequences"):
            print(f"\nProcessing batch {i+1}/{args.n_batches}")
            sequences = main(args.ec_label, model, special_tokens, device, tokenizer, 
                            args.plddt_threshold, args.perplexity_threshold, 
                            args.min_length, args.max_length, args.use_pivot, 
                            args.n_continuations, args.control_tag)
            if args.ec_label in sequences:
                all_sequences[args.ec_label].extend(sequences[args.ec_label])
                
            # Clear GPU memory after each batch
            torch.cuda.empty_cache()

        if args.data_type == "train":
            output_dir = f"train_data_iteration{args.iteration_num}" + (f"_{args.tag}" if args.tag else "")
        else:  # val
            output_dir = f"val_data" + (f"_{args.tag}" if args.tag else "")

        # Make this directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save sequences to file 
        with open(f"{output_dir}/sequences_{args.control_tag.replace('<', '').replace('>', '')}.json", "w") as f:
            json.dump(all_sequences, f)
    
    if args.data_type == "train":
        output_dir = f"train_data_iteration{args.iteration_num}" + (f"_{args.tag}" if args.tag else "")
    else:  # val
        output_dir = f"val_data" + (f"_{args.tag}" if args.tag else "")

    # Sort sequences by length for each EC label
    for ec_label in all_sequences:
        all_sequences[ec_label].sort(key=lambda x: len(x[0]))  # Sort by sequence length (x[0] is the sequence string)

    # Process sequences and add stability scores
    print("\nComputing stability scores...")
    
    df = process_sequences_with_stability(all_sequences, args.plddt_threshold, 
                                          args.perplexity_threshold, args.min_length, args.max_length,
                                          disable_filtering=args.disable_filtering, scoring_method=args.scoring_method)
    
    # Save results
    print("\nSaving results...")
    save_data(df, output_dir=output_dir)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print("Data generation completed!") 