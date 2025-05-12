import csv
import torch
from tqdm import tqdm
import argparse

from stability import stability_score, stability_score_batch
from utils import perplexity_from_logits

from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

def generate_stability_labels_from_hf(ec_number, output_path, limit=None):
    """
    Load sequences from HuggingFace dataset, add stability score, pLDDT, and perplexity columns,
    and save to a new CSV file.
    
    Args:
        ec_number (str): EC number to filter sequences.
        output_path (str): Path to the output CSV file.
        limit (int, optional): Limit the number of sequences to process.
    """
    
    ds = load_dataset("AI4PD/ZymCTRL")
    filtered_ds = ds.filter(lambda x: x['text'].startswith(ec_number))
    
    # Load the model and tokenizer for perplexity calculation
    model_name = 'AI4PD/ZymCTRL'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    rows = []
    total_items = len(filtered_ds['train']) if limit is None else min(limit, len(filtered_ds['train']))
    
    # Process in batches
    batch_size = 8  # Adjust based on available GPU memory
    current_batch = []
    current_batch_indices = []
    
    for i, item in tqdm(enumerate(filtered_ds['train']), total=total_items, desc="Processing sequences"):
        if limit is not None and i >= limit:
            break
            
        try:
            # Extract the sequence from the 'text' field
            text = item['text']
            ec_number, sequence = text.split('<sep>')[:2]
            sequence = sequence.split('<start>')[1].split('<end>')[0]
            
            current_batch.append(sequence)
            current_batch_indices.append((i, ec_number.strip()))
            
            # Process batch when it's full or on the last item
            if len(current_batch) == batch_size or i == total_items - 1:
                # Calculate stability and pLDDT for the batch
                stability_results = stability_score_batch(current_batch)
                
                # Calculate perplexity for the batch
                batch_input_ids = tokenizer(current_batch, padding=True, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    outputs = model(batch_input_ids.input_ids)
                    batch_ppls = perplexity_from_logits(outputs.logits, batch_input_ids.input_ids, batch_input_ids.attention_mask)
                
                # Process results for each sequence in the batch
                for (idx, ec), seq, (raw_if, dg, plddt), ppl in zip(current_batch_indices, current_batch, stability_results, batch_ppls):
                    if dg < -2.0:
                        stability_label = "high"
                    elif dg > 0.0:
                        stability_label = "low"
                    else:
                        stability_label = "medium"
                    
                    rows.append([
                        ec,
                        seq,
                        str(raw_if),
                        str(dg),
                        stability_label,
                        str(plddt),
                        str(ppl.item()),
                        str(len(seq))
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
    
    # Write to output CSV
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['EC_Number', 'Sequence', 'Raw_IF', 'DeltaG', 'Stability_Label', 'pLDDT', 'Perplexity', 'Length'])
        writer.writerows(rows)
    
    print(f"Processed {len(rows)} sequences and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stability labels from HuggingFace dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sequences to process")
    args = parser.parse_args()

    generate_stability_labels_from_hf("4.2.1.1", "brenda/ec_sequences_stability_plddt_perplexity.csv", limit=args.limit)