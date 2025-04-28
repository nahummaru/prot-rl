import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def validate_sequence(sequence):
    """
    Validate that the sequence contains only canonical amino acids
    """
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(char in canonical_amino_acids for char in sequence)

def generate_sequences(model, tokenizer, ec_label, device, num_sequences=20):
    """
    Generate sequences with EC tag
    """
    sequences = []
    input_text = f"{ec_label}<sep><start>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    for i in range(num_sequences):
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
        
        # Clean up the sequence
        sequence = sequence.replace('<start>', '').replace('<end>', '').strip()
        
        # Validate sequence
        if validate_sequence(sequence):
            sequences.append(sequence)
            print(f"Generated valid sequence {i+1}/{num_sequences}: {sequence[:20]}...")
        else:
            print(f"Generated invalid sequence {i+1}/{num_sequences} (skipped)")
    
    return sequences

def save_sequences(sequences, ec_label, iteration_num, output_dir="test_output"):
    """
    Save generated sequences to a FASTA file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/test_seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    
    with open(output_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">test_{ec_label}_{i}_iteration{iteration_num}\n{seq}\n")
    
    print(f"Saved {len(sequences)} sequences to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, default=0)
    parser.add_argument("--ec_label", type=str, default="4.2.1.1")
    parser.add_argument("--num_sequences", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="AI4PD/ZymCTRL")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    
    # Generate sequences
    print(f"Generating {args.num_sequences} sequences for EC {args.ec_label}")
    sequences = generate_sequences(model, tokenizer, args.ec_label, device, args.num_sequences)
    
    # Save sequences
    save_sequences(sequences, args.ec_label, args.iteration_num)
    
    print("Sequence generation test completed successfully")

if __name__ == "__main__":
    main() 