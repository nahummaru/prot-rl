import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from stability import compute_stability
import pandas as pd
import numpy as np

def evaluate_multi_tag_performance(model, tokenizer, ec_label, device):
    """
    Evaluate model's performance with multiple control tags
    """
    results = {}
    
    # Test different tag combinations
    tag_combinations = [
        {"stability": "high"},
        {"stability": "high", "solubility": "high"},
        {"stability": "low"},
        {"stability": "low", "solubility": "low"}
    ]
    
    for tags in tag_combinations:
        # Generate sequences with specific tags
        sequences = generate_sequences(model, tokenizer, ec_label, device, tags)
        
        # Compute stability scores
        stability_scores = [compute_stability(seq) for seq in sequences]
        
        # Compute tag adherence
        tag_adherence = compute_tag_adherence(sequences, tags)
        
        # Store results
        tag_key = "_".join([f"{k}={v}" for k, v in tags.items()])
        results[tag_key] = {
            'stability_mean': np.mean(stability_scores),
            'stability_std': np.std(stability_scores),
            'tag_adherence': tag_adherence
        }
    
    return results

def generate_sequences(model, tokenizer, ec_label, device, tags, num_sequences=20):
    """
    Generate sequences with multiple control tags
    """
    sequences = []
    tag_str = " ".join([f"<{k}={v}>" for k, v in tags.items()])
    input_text = f"{ec_label}<sep><start>{tag_str}"
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

def compute_tag_adherence(sequences, tags):
    """
    Compute how well sequences adhere to specified tags
    """
    adherence_scores = []
    
    for seq in sequences:
        # For stability tag
        if "stability" in tags:
            stability_score = compute_stability(seq)
            if tags["stability"] == "high":
                adherence = 1 if stability_score > 0.7 else 0
            else:
                adherence = 1 if stability_score < 0.3 else 0
            adherence_scores.append(adherence)
    
    return np.mean(adherence_scores)

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
    results = evaluate_multi_tag_performance(model, tokenizer, args.ec_label, device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv(f"{args.output_dir}/multi_tag_eval_iteration{args.iteration_num}.csv")
    
    print(f"Multi-tag evaluation results saved to {args.output_dir}/multi_tag_eval_iteration{args.iteration_num}.csv")

if __name__ == "__main__":
    main() 