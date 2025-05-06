import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from stability import compute_stability
import pandas as pd
import numpy as np
import json

def evaluate_ec_capabilities(model, tokenizer, ec_label, device):
    """
    Evaluate model's EC-specific generation capabilities
    """
    # Generate sequences with EC tag only
    sequences = generate_sequences(model, tokenizer, ec_label, device)
    
    # Compute stability scores
    stability_scores = [compute_stability(seq) for seq in sequences]
    
    # Compute perplexity
    perplexities = compute_perplexity(model, tokenizer, sequences, device)
    
    return {
        'stability_mean': np.mean(stability_scores),
        'stability_std': np.std(stability_scores),
        'perplexity_mean': np.mean(perplexities),
        'perplexity_std': np.std(perplexities)
    }

def generate_sequences(model, tokenizer, ec_label, device, num_sequences=20):
    """
    Generate sequences with EC tag
    """
    sequences = []
    input_text = f"{ec_label}<sep><start>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    for _ in range(num_sequences):
        outputs = model.generate(
            input_ids,
            max_length=1024,
            do_sample=True,
            top_k=9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sequences.append(sequence)
    
    return sequences

def brenda_sequences(brenda_path, ec_label, num_sequences=20):
    try:
        # Load the BRENDA database CSV file
        df = pd.read_csv(brenda_path)
        
        # Filter sequences by EC label
        filtered_df = df[df['EC_NUMBER'] == ec_label]
        
        if filtered_df.empty:
            print(f"No sequences found for EC label: {ec_label}")
            return []
        
        sequences = filtered_df['sequence'].tolist()
        
        # Return up to num_sequences
        return sequences[:num_sequences]
    
    except Exception as e:
        print(f"Error loading BRENDA sequences: {e}")
        return []

def compute_perplexity(model, tokenizer, sequences, device):
    """
    Compute perplexity of generated sequences
    """
    perplexities = []
    for seq in sequences:
        input_ids = tokenizer.encode(seq, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return perplexities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--ec_label", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_name = f"output_iteration{args.iteration_num}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Evaluate
    results = evaluate_ec_capabilities(model, tokenizer, args.ec_label, device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(f"{args.output_dir}/ec_eval_iteration{args.iteration_num}.csv", index=False)
    
    print(f"EC evaluation results saved to {args.output_dir}/ec_eval_iteration{args.iteration_num}.csv")

if __name__ == "__main__":
    main() 