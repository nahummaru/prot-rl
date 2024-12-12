from torch.nn import CrossEntropyLoss
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import random
import numpy as np
from functools import partial
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
import torch.optim as optim
import statistics
import json
import esm

# HYPERPARAMETERS
beta = 0.01
seed = 1010
learning_rate = 1e-7
batch_size = 5
num_epochs = 5
count = 0
split_percent = 0.2  # of the eval set
beta1 = 0.9
beta2 = 0.98
epsilon = 1e-8
adam_decay = 0.1

target_length=621

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)



def extract_sequences(rep_seq):
    sequences_rep = dict()
     
    for line in rep_seq:
            if ">" in line:
                name = line.replace(">", "").replace("\n", "")
                pprplx = line.split("\t")[1]
                emb_identifier = line.replace(">", "").replace("\n", "")
            else:
                aa = line.strip().split(':')[1]
                # Retrieve pLA, ipTM and ptm from Alpha fold prediction:
                try:
                    iptm, pae_interaction, plddt, ptm = af2_metrics(name,target_length, iteration_num)
                    print(name)
                    sequences_rep[name] = {
                                "sequence" : aa,
                                'iptm'  : float(iptm),
                                'pae_interation'  : float(pae_interaction),
                                'plddt'  : plddt,
                                'ptm'  : float(ptm),
                                    }
                except:
                    print(f'WARNING for seq {name}')
    return sequences_rep


def generate_dataset(iteration_num, ec_label):
     data = dict()
     data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }
    
     with open(f"seq_gen_{ec_label}_iteration{iteration_num-1}.fasta", "r") as f:
        rep_seq = f.readlines()
     
     
     # Get the amminoacid sequences 
     sequences_rep = extract_sequences(rep_seq)
     
     # Calculate esm log-likelihood
     sequences_rep = compute_pll(sequences_rep)                               

     for entry in sequences_rep:
            sequence = sequences_rep[entry]['sequence']
            lenght_rew = math.exp(-((((len(sequence)/50)-1)**2)/(0.5**2))) # Gaussian center on 1. The closer the ratio between len and aligment is, the higher is the reward
            esm_ppl = sequences_rep[entry]['emb']
            pae = sequences_rep[entry]['pae_interation']
            iptm = sequences_rep[entry]['iptm']
            
            print(f'Scores for: {entry}: pae: {pae}: iptm: {iptm} : esm_ppl: {esm_ppl}: sequence: {sequence}')
            # Save in a dataset
            data["sequence"].append(formatting_sequence(sequence, ec_label))
            data["seq_name"].append(entry)
            data["weight"].append(((-pae)+(iptm*10)+(-esm_ppl/100))*(lenght_rew))
     
     hf_dataset = Dataset.from_pandas(pd.DataFrame(data))
     shuffled_dataset = hf_dataset.shuffle(seed=seed)
     # Split the dataset (80% train, 20% eval)
     train_size = int((1-split_percent) * len(shuffled_dataset))
     train_dataset = shuffled_dataset.select(range(train_size))
     eval_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))
  
     # Create a DatasetDict to hold the train and eval datasets
     final_dataset = DatasetDict({
          'train': train_dataset,
          'eval': eval_dataset
          })
          
     final_dataset.save_to_disk(f"dataset_iteration{iteration_num}")
     

     return final_dataset
     
def af2_metrics(name, target_length, iteration_num):
    output_folder = f"AF/iteration{iteration_num-1}"
    files = os.listdir(output_folder)
    name = name.replace('\t','_') 
    pae_file = f"{name}_predicted_aligned_error_v1.json"
    
    with open(os.path.join(output_folder, pae_file), "r") as f:
        pae = np.array(json.load(f)["predicted_aligned_error"])
    
    binder_len = (
        len(pae) - target_length
    )  
    pae_interaction = (
        pae[:binder_len, binder_len:].mean() + pae[binder_len:, :binder_len].mean()
    ) / 2


    for file in files:
        if file.startswith(f"{name}_scores_rank_001"):
            with open(os.path.join(output_folder, file), "r") as f:
                metrics = json.load(f)
            plddt = metrics['plddt']
            ptm = metrics['ptm']
            iptm = metrics['iptm']
    
    return iptm, pae_interaction, plddt, ptm


def compute_pll(sequences):

    # Load the ESM model and tokenizer
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    
    for x in sequences:
      sequence = sequences[x]['sequence']
      data = [("protein", sequence)]
      batch_converter = alphabet.get_batch_converter()
      *_, batch_tokens = batch_converter(data)
      log_probs = []
      for i in range(len(sequence)):
          batch_tokens_masked = batch_tokens.clone()
          batch_tokens_masked[0, i + 1] = alphabet.mask_idx
          with torch.no_grad():
              token_probs = torch.log_softmax(
                  model(batch_tokens_masked.to(device))["logits"], dim=-1
              )
          log_probs.append(token_probs[0, i + 1, alphabet.get_idx(sequence[i])].item())
      sequences[x]['emb'] = float(math.fsum(log_probs))
    
    del model
    
    return sequences


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def formatting_sequence(sequence, ec_label):
    sequence = str(f"{ec_label}<sep><start>{sequence}<end><|endoftext|>")
    return sequence
    


def log_likelihood(sequences, device, model, tokenizer):
    
    all_loss = []  # List to store loss for each sequence

    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
        outputs = model(inputs, labels=inputs)
        loss, logits = outputs[:2]
        all_loss.append(loss.unsqueeze(0))
        
    all_loss = torch.cat(all_loss)
    
    return all_loss

def dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta=0.1):

    if ref_log_probs is None:
        log_ratios = beta * policy_log_probs
    else:
        log_ratios = (beta * (policy_log_probs.to(device) - ref_log_probs.to(device)))
    weights = torch.softmax(weights*(-1), dim=0)

    return F.cross_entropy(log_ratios, weights)

# Training function
def train(model, ref_model, tokenizer, train_loader, optimizer, device):
    model.train()
    total_loss = []
    for batch in train_loader:
        optimizer.zero_grad()

        sequences = batch["sequence" ]  
        ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
        policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
        weights = batch["weight"].to(device)
        
        # Calculate DPO loss
        loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        print(f'loss:{loss}')
        total_loss.append(loss.item())


    return sum(total_loss) / len(total_loss)



def evaluate(model,ref_model,tokenizer, eval_loader, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in eval_loader:
            sequences = batch["sequence" ]  
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
            weights = batch["weight"].to(device)
            
            # Calculate DPO loss
            loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta)
    
            total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer for sequence generation.
    """
    try:
        # Check if the output directory exists; create it if it does not
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Model and tokenizer saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred while saving the model and tokenizer: {e}")





def main(train_loader,eval_loader, iteration_num):
  # Load the model
  
  model_name = f"output_iteration{iteration_num-1}" # here model 
  
  print(f'Model {model_name} has been loaded')

  tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  ref_model = AutoModelForCausalLM.from_pretrained('output_iteration0').to(device)
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=adam_decay)



  for epoch in range(num_epochs):
      train_loss = train(model, ref_model, tokenizer, train_loader, optimizer, device)
      eval_loss = evaluate(model, ref_model, tokenizer, eval_loader, device)
  
      print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
      
      save_model_and_tokenizer(model, tokenizer, output_dir=f"output_iteration{iteration_num}")

  del model
  del ref_model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.label
    ec_label = ec_label.strip()
    seed_everything(seed)
    
    if not os.path.exists(f"dataset_iteration{iteration_num}"):
      dataset = generate_dataset(iteration_num, ec_label)
    else:
      dataset = load_from_disk(f"dataset_iteration{iteration_num}")
    
    print('Dataset Loaded!!')
    train_set = dataset["train"]
    eval_set = dataset["eval"]


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    main(train_loader,eval_loader, iteration_num)

