"""
This file defines the following evals:
- PPL of BRENDA sequences
- Controlability of generated sequence stability
"""

from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..stability import stability_score

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

@torch.no_grad()
def get_ppl(model, input_ids):
    """
    Compute perplexity for input sequence without gradient tracking
    Input_ids should be a list of tensors
    """
    ppls = []
    for input_id in tqdm(input_ids, desc="Computing PPL"):
        outputs = model(input_id, labels=input_id)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        ppls.append(perplexity)
    return ppls

def generate_sequences(model, tokenizer, ec_label, device, num_sequences=20, control_tag=None):
    """
    Generate sequences with EC tag
    """
    sequences = []
    input_text = f"{ec_label}<sep><start>"
    if control_tag:
        input_text = f"{ec_label}<stability={control_tag}><sep><start>"
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

def main():
  target_model_name = "AI4PD/ZymCTRL"
  model_name = "AI4PD/ZymCTRL"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
  target_model = None # TODO: load target model somehow

  brenda_path = "DPO_ZymCTRL/data/brenda_enzymes.csv"
  ec_label = "4.2.1.1"
  num_sequences = 20

  brenda_sequences = brenda_sequences(brenda_path, ec_label, num_sequences=num_sequences)

  # Compare PPL
  base_ppls = get_ppl(base_model, tokenizer, brenda_sequences)
  target_ppls = get_ppl(target_model, tokenizer, brenda_sequences)

  plt.hist(base_ppls, bins=10, alpha=0.5, label='Base Model')
  plt.hist(target_ppls, bins=10, alpha=0.5, label='Target Model')
  plt.legend()
  plt.show()
  plt.savefig('ppl_dist_comparison.png')

  # Compare controlability

  target_sequences = {
      "high": [],
      "medium": [],
      "low": []
  }

  base_sequences = []

  # Sample sequences from base model
  for i in tqdm(range(num_sequences), desc="Generating base sequences"):
    gen_sequence = generate_sequences(base_model, tokenizer, ec_label, device, num_sequences=1)
    base_sequences.append(gen_sequence)

  # Sample sequences from target model
  for control_tag in target_sequences.keys():
    for i in tqdm(range(num_sequences), desc=f"Generating {control_tag} stability sequences"):
      gen_sequence = generate_sequences(target_model, tokenizer, ec_label, device, num_sequences=1, control_tag=control_tag)
      target_sequences[control_tag].append(gen_sequence)

  # Calculate controlability for target and base models 
  target_stability_scores = {
      "high": [],
      "medium": [],
      "low": []
  }

  base_stability_scores = []

  # TODO: perhaps swap out for vectorized stability_score impl

  for seq in tqdm(base_sequences, desc="Computing base stability scores"):
    base_stability_scores.append(stability_score([seq]))

  for control_tag in target_sequences.keys():
    for seq in tqdm(target_sequences[control_tag], desc=f"Computing {control_tag} stability scores"):
      target_stability_scores[control_tag].append(stability_score([seq]))

  plt.hist(base_stability_scores, label="Base Model (uncontrolled)")
  plt.hist(target_stability_scores["high"], label="Target Model (high)")
  plt.hist(target_stability_scores["medium"], label="Target Model (medium)")
  plt.hist(target_stability_scores["low"], label="Target Model (low)")

  plt.legend()
  plt.show()
  plt.savefig('stability_dist_comparison.png')

  # Calculate KL divergence between distributions
  # from scipy.stats import entropy
  # import numpy as np

  # def calculate_kl_divergence(p, q):
  #     """Calculate KL divergence between two distributions"""
  #     # Add small epsilon to avoid log(0)
  #     epsilon = 1e-10
  #     p = np.array(p) + epsilon
  #     q = np.array(q) + epsilon
      
  #     # Normalize distributions
  #     p = p / np.sum(p)
  #     q = q / np.sum(q)
      
  #     return entropy(p, q)

  # # Create histograms for each distribution
  # base_hist, base_bins = np.histogram(base_stability_scores, bins=20, density=True)
  # high_hist, _ = np.histogram(target_stability_scores["high"], bins=base_bins, density=True)
  # medium_hist, _ = np.histogram(target_stability_scores["medium"], bins=base_bins, density=True)
  # low_hist, _ = np.histogram(target_stability_scores["low"], bins=base_bins, density=True)

  # # Calculate KL divergences
  # kl_high = calculate_kl_divergence(high_hist, base_hist)
  # kl_medium = calculate_kl_divergence(medium_hist, base_hist)
  # kl_low = calculate_kl_divergence(low_hist, base_hist)

  # print(f"KL Divergence (High vs Base): {kl_high:.4f}")
  # print(f"KL Divergence (Medium vs Base): {kl_medium:.4f}")
  # print(f"KL Divergence (Low vs Base): {kl_low:.4f}")

  # # Plot KL divergences
  # plt.figure(figsize=(10, 6))
  # stability_levels = ['High', 'Medium', 'Low']
  # kl_values = [kl_high, kl_medium, kl_low]
  
  # plt.bar(stability_levels, kl_values)
  # plt.title('KL Divergence from Base Distribution')
  # plt.ylabel('KL Divergence')
  # plt.savefig('kl_divergence_comparison.png')
  # plt.close()