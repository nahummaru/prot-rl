import csv
import torch
from tqdm import tqdm
import argparse
import os

from stability import stability_score_batch

from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

def create_test_sequences():
    """
    Create 3 test sequences for testing the scoring functionality.
    Returns a list of (sequence, length) tuples.
    """
    test_sequences = [
        # Short stable protein-like sequence
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGSTLPRALKFKQIQNPDIKEELKKAIGIKIDKDEKAAALKAINEELAELTEEEAKKMLNQALEATLHEILAEEIKATKVDAKAAAEGKVDGKVKELRKVPGVTLVGGVPKDDFIEKFKLHPEEEGKLTEGFCKVLKGVADEVHEKGDVEGVAVGVTKDFDAKLEHHHHHH",
        
        # Medium length sequence
        "MTMDLLITPEQAELLAKWLETKPPSGKAVFGLMGTGGGGTGSTGSLPTESEIIAIDGSGFGPVGAYGVGGVGGQGLGLVGQGVEAYALKYKVEAQALGSQAKLKGLLETKPPSGKAVFGLMGTGGGGTGSTGSLPTESEIIAIDGSGFGPVGAYGVGGVGGQGLGLVGQGVEAYALKYKVEAQALGSQAKLKGLL",
        
        # Longer more complex sequence
        "MKTEVEFSHEYWMRHALTQETYLQRVNRLHYIYNFLHTVQHQGSAYWQRYSLNALSPYDAATNVQIGAGLYDVDLSPDWKQYGIQKEAIEVWGGLAKSLDELAKSGGIGREYATFRGATIEADGFVPHADRWVLRMVKSLKDLKKSAPSLKGLFAYEAPGLKPERADIVSQFGGKPRVFGFGAIIGEPGTGKTLASVDPSNLKQRQNRKTIPAYKQALGELGAHPNIQDRPQEMQAWIDRGNWDAFGKGRRTASMGKGILTNQAAYLPAFDETFGVDKLGNIWRYHELTQNAALLQSWFSDFLHGLKDYIPFTDFFDPMGTANAELEDIFGVKASDAKPKALWEAVKRHGYQFIAGSYDRQKFTQDLIARNLAIGSVITVAGWCPNLDFGYYPDEFGRLTIASDPRVTVRQPKYGPQTPSIRLMPVNAIAQGPGLKGGNPKQYIGHLQVLLLDTRRNLQIDLQSDEEAAQKLGLTRPVLMGFAFGNGTDKLTSSFASCLEFAYEFAYSFVAMGFGHLRTNASAQFEGQKSQLGPQKPVGNLPFIEMFTLQEQNASMLSTDIIKQLSEFPKEGTGQNFDLNVNKDGDAYDYFDMDPGQAKFFNRVDYEFIKGDFGFDGFNAPIYFPSRFKMLQGKLREWFSEVSLLKAGKGQPKEWEFMPAFGFLQRMHKLPKTKDVDGYVFTPLSRFKMLQGKLREWFSEVSLLKAGKGQP"
    ]
    
    return [(seq, len(seq)) for seq in test_sequences]

def generate_stability_labels_test(output_path, scoring_method="rosetta"):
    """
    Generate stability labels for test sequences without downloading dataset.
    
    Args:
        output_path (str): Path to the output CSV file.
        scoring_method (str): Scoring method for stability evaluation ('rosetta' or 'esm-if').
    """
    
    print(f"Running in TEST MODE with 3 fake sequences")
    print(f"Using {scoring_method.upper()} scoring method")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Get test sequences
    test_data = create_test_sequences()
    sequences = [seq for seq, length in test_data]
    
    print(f"Processing {len(sequences)} test sequences...")
    
    # Process all sequences in one batch since we only have 3
    batch_size = len(sequences)
    
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['EC_Number', 'Sequence', 'Raw_Score', 'DeltaG', 'Stability_Label', 'pLDDT', 'Length', 'Scoring_Method'])
        
        try:
            print(f"Calculating stability scores using {scoring_method}...")
            
            # Calculate stability and pLDDT for the batch using specified scoring method
            stability_results = stability_score_batch(sequences, scoring_method=scoring_method)
            
            # Process results for each sequence
            for i, (seq, (raw_score, dg, plddt)) in enumerate(zip(sequences, stability_results)):
                # Assign stability labels based on scoring method
                if scoring_method == "rosetta":
                    # Rosetta energy units: more negative = more stable
                    if dg < -50:  # Very stable
                        stability_label = "high"
                    elif dg > 0:   # Unstable
                        stability_label = "low"
                    else:          # Moderately stable
                        stability_label = "medium"
                else:  # esm-if
                    # ESM-IF converted deltaG: more negative = more stable
                    if dg < -2.0:
                        stability_label = "high"
                    elif dg > 0.0:
                        stability_label = "low"
                    else:
                        stability_label = "medium"
                
                print(f"Sequence {i+1}: Length={len(seq)}, Raw_Score={raw_score:.2f}, ΔG={dg:.2f}, Label={stability_label}, pLDDT={plddt:.1f}")
                
                writer.writerow([
                    "4.2.1.1",  # Default EC number for test
                    seq,
                    str(raw_score),
                    str(dg),
                    stability_label,
                    str(plddt),
                    str(len(seq)),
                    scoring_method
                ])
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing test sequences: {str(e)}")
            raise e
    
    print(f"Test completed! Results saved to {output_path}")

def generate_stability_labels_from_hf(ec_number, output_path, limit=None, min_length=180, max_length=300, scoring_method="rosetta"):
    """
    Load sequences from HuggingFace dataset, add stability score, pLDDT, and perplexity columns,
    and save to a new CSV file.
    
    Args:
        ec_number (str): EC number to filter sequences.
        output_path (str): Path to the output CSV file.
        limit (int, optional): Limit the number of sequences to process.
        min_length (int, optional): Minimum length of sequences to process.
        max_length (int, optional): Maximum length of sequences to process.
        scoring_method (str): Scoring method for stability evaluation ('rosetta' or 'esm-if').
    """
    
    print(f"Loading sequences for EC number: {ec_number}")
    print(f"Using {scoring_method.upper()} scoring method")
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

    filtered_ds = filtered_ds.filter(lambda x: len(x['text']) <= max_length)
    filtered_ds = filtered_ds.filter(lambda x: len(x['text']) >= min_length)
    filtered_ds = filtered_ds.map(lambda x: {'length': len(x['text'])})
    filtered_ds = filtered_ds.sort('length', reverse=False)
    
    print(f"Found {len(filtered_ds['train'])} sequences after filtering")
    
    # Load the model and tokenizer for perplexity calculation (if needed)
    # model_name = 'AI4PD/ZymCTRL'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer.pad_token = tokenizer.eos_token

    total_items = len(filtered_ds['train']) if limit is None else min(limit, len(filtered_ds['train']))
    
    # Process in batches
    batch_size = 8  # Reduced batch size for stability due to memory constraints
    current_batch = []
    current_batch_indices = []
    
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['EC_Number', 'Sequence', 'Raw_Score', 'DeltaG', 'Stability_Label', 'pLDDT', 'Length', 'Scoring_Method'])
        
        for i, item in tqdm(enumerate(filtered_ds['train']), total=total_items, desc="Processing sequences"):
            if limit is not None and i >= limit:
                break
                
            try:
                # Extract the sequence from the 'text' field
                sequence = item['text']
                
                current_batch.append(sequence)
                current_batch_indices.append((i, ec_number.strip()))
                
                # Process batch when it's full or on the last item
                if len(current_batch) == batch_size or i == total_items - 1 or i == len(filtered_ds['train']) - 1:
                    print(f"Processing batch of {len(current_batch)} sequences...")
                    
                    # Calculate stability and pLDDT for the batch using specified scoring method
                    stability_results = stability_score_batch(current_batch, scoring_method=scoring_method)
                    
                    # Process results for each sequence in the batch
                    for (idx, ec), seq, (raw_score, dg, plddt) in zip(current_batch_indices, current_batch, stability_results):
                        # Assign stability labels based on scoring method
                        if scoring_method == "rosetta":
                            # Rosetta energy units: more negative = more stable
                            if dg < -50:  # Very stable
                                stability_label = "high"
                            elif dg > 0:   # Unstable
                                stability_label = "low"
                            else:          # Moderately stable
                                stability_label = "medium"
                        else:  # esm-if
                            # ESM-IF converted deltaG: more negative = more stable
                            if dg < -2.0:
                                stability_label = "high"
                            elif dg > 0.0:
                                stability_label = "low"
                            else:
                                stability_label = "medium"
                        
                        writer.writerow([
                            ec,
                            seq,
                            str(raw_score),
                            str(dg),
                            stability_label,
                            str(plddt),
                            str(len(seq)),
                            scoring_method
                        ])
                    
                    # Clear batch
                    current_batch = []
                    current_batch_indices = []
                    torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"Error processing entry {i}: {str(e)}")
                print(f"Skipping entry {i}")
                # Clear batch on error to prevent accumulation of problematic sequences
                current_batch = []
                current_batch_indices = []
    
    print(f"Processed {total_items} sequences and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stability labels from HuggingFace dataset or test sequences")
    parser.add_argument("--test", action="store_true", help="Run in test mode with 3 fake sequences instead of downloading dataset")
    parser.add_argument("--ec_number", type=str, default="4.2.1.1", help="EC number to filter sequences")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sequences to process")
    parser.add_argument("--min_length", type=int, default=180, help="Minimum length of sequences to process")
    parser.add_argument("--max_length", type=int, default=300, help="Maximum length of sequences to process")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--scoring_method", type=str, default="rosetta", choices=["esm-if", "rosetta"],
                      help="Scoring method for stability evaluation: 'rosetta' (default) or 'esm-if'")
    args = parser.parse_args()

    if args.test:
        # Run test mode with fake sequences
        generate_stability_labels_test(args.output_path, args.scoring_method)
    else:
        # Run normal mode with HuggingFace dataset
        generate_stability_labels_from_hf(
            args.ec_number, 
            args.output_path, 
            limit=args.limit, 
            min_length=args.min_length, 
            max_length=args.max_length,
            scoring_method=args.scoring_method
        )