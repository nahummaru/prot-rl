import torch, os, math, argparse
from transformers import GPT2LMHeadModel, AutoTokenizer
from stability import stability_score
import pandas as pd
from tqdm import tqdm
import json

def remove_characters(sequence, char_list):
    '''
    Removes special tokens used during training.
    '''
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids, model, tokenizer):
    '''
    Computes perplexities for the generated sequences. 
    '''
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model, special_tokens, device, tokenizer):
    '''
    Function to generate sequences from the loaded model.
    '''
    print(f"Generating sequences for label: {label}")
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    
    # Generating sequences
    outputs = model.generate(
        input_ids, 
        top_k=9, 
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=2)  # Using smaller batch size
    
    print(f"Generated {len(outputs)} raw sequences")
    
    # Check sequence sanity, ensure sequences are not-truncated
    new_outputs = [output for output in outputs if output[-1] == 0]
    
    if not new_outputs:
        print("not enough sequences with short lengths!!")
        return {}

    print(f"After filtering: {len(new_outputs)} valid sequences")

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) 
            for output in new_outputs]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) 

    # Final results dictionary without strange characters
    sequences = {}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

def process_sequences_with_stability(sequences_dict):
    """
    Add stability scores to sequences and convert to DataFrame.
    Processes sequences in batches using stability_score.
    """
    data = []
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Collect valid sequences for batch processing
    valid_sequences = []
    sequence_metadata = []  # Store corresponding metadata (label, perplexity)
    
    for label, seq_list in sequences_dict.items():
        for seq, ppl in seq_list:
            # Check if sequence is valid
            if not all(c in canonical_amino_acids for c in seq):
                print(f"Skipping invalid sequence: {seq[:20]}...")
                continue
            
            valid_sequences.append(seq)
            sequence_metadata.append((label, ppl))
    
    print(f"\nComputing stability for {len(valid_sequences)} sequences...")
    # Get stability scores for all sequences at once
    stability_results = stability_score(valid_sequences)
    
    # Process results
    for (seq, (label, ppl)), (raw_if, dg) in zip(zip(valid_sequences, sequence_metadata), stability_results):
        # Use deltaG as the stability score
        stability = dg
        
        # Assign stability label based on deltaG
        # Note: Adjusting thresholds since deltaG is in kcal/mol
        if stability < -2.0:  # More negative = more stable
            stability_label = "high"
        elif stability > 0.0:
            stability_label = "low"
        else:
            stability_label = "medium"
            
        data.append({
            'sequence': seq,
            'perplexity': ppl,
            'stability_score': stability,
            'stability_raw_if': raw_if,
            'stability_label': stability_label,
            'ec_label': label
        })
    
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

    df.to_csv(f"{output_dir}/sequences.csv", index=False)
    
    # Save FASTA files by stability label
    for label in ['high', 'low', 'medium']:
        subset = df[df['stability_label'] == label]
        with open(f"{output_dir}/stability_{label}.fasta", 'w') as f:
            for idx, row in subset.iterrows():
                f.write(f">{row['ec_label']}_{idx}_stability={label}_score={row['stability_score']:.2f}\n")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    if args.model_path:
        model_name = args.model_path
    elif args.iteration_num == 0:
        model_name = 'AI4PD/ZymCTRL'
    else:
        model_name = f'output_iteration{args.iteration_num}'
    
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    # Generate sequences in batches
    all_sequences = {}
    all_sequences[args.ec_label] = []
    
    print(f"Generating {args.n_batches} batch(es) of sequences")
    for i in range(args.n_batches):
        print(f"\nProcessing batch {i+1}/{args.n_batches}")
        sequences = main(args.ec_label, model, special_tokens, device, tokenizer)
        if args.ec_label in sequences:
            all_sequences[args.ec_label].extend(sequences[args.ec_label])
            
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()

    output_dir = f"training_data_iteration{args.iteration_num}" + (f"_{args.tag}" if args.tag else "")

    # Save sequences to file 
    with open(f"{output_dir}/sequences.json", "w") as f:
        json.dump(all_sequences, f)
    
    # Process sequences and add stability scores
    print("\nComputing stability scores...")
    df = process_sequences_with_stability(all_sequences)
    
    # Save results
    print("\nSaving results...")
    save_data(df, output_dir=output_dir)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print("Data generation completed!") 