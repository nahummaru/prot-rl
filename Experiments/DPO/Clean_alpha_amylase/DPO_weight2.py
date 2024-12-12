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
import glob
from Bio.PDB import PDBParser

# HYPERPARAMETERS
beta = 0.01
seed = 1998
learning_rate = 1e-7
batch_size = 5
num_epochs = 5
count = 0
split_percent = 0.2  # of the eval set
beta1 = 0.9
beta2 = 0.98
epsilon = 1e-8
adam_decay = 0.1


device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


class LayerNormNet(nn.Module):
    '''
    Class from CLEAN https://github.com/tttianhao/CLEAN
    Tianhao Yu et al., Enzyme function prediction using contrastive 
    learning.Science379,1358-1363(2023).DOI:10.1126/science.adf2465
    '''
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def analyse_pdb(sequences_rep, iteration_num):
    pdb_folder = f'output_iteration{iteration_num-1}/PDB'
    pdb_files = glob.glob(os.path.join(pdb_folder, '*.pdb'))
    print(f'there are {len(pdb_files)} PDBs')
    for pdb in pdb_files:
        name = os.path.basename(pdb).split('.pdb')[0]
        plddt = extract_mean_plddt(pdb)
        sequences_rep[name]['plddt'] = plddt
        print(plddt)
    return sequences_rep

def extract_mean_plddt(pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb)
    b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
    if b_factors:
        plddt = sum(b_factors) / len(b_factors)
    else:
        plddt = 0
    return plddt

def retieve_activity_prediction(sequences_rep, iteration, ec_label):
    with open(f'activity_prediction_iteration{iteration_num-1}.txt') as f:
        rep_seq = f.readlines()
    for x in rep_seq:
        name, activity = x.split(',')
        name = name.split('\t')[0]
        sequences_rep[name[1:]]['activity'] = float(activity)
    return sequences_rep

def generate_dataset(iteration_num, ec_label):
     data = dict()
     data = {
        "sequence" : [],
        "seq_name" : [],
        "TM" : [],
        "TM_norm_que" : [],
        "weight" : []
        }
    
     with open(f"seq_gen_{ec_label}_iteration{iteration_num-1}.fasta", "r") as f:
        rep_seq = f.readlines()

     with open(f"{ec_label}_TM_iteration{iteration_num-1}", "r") as f:
        alpha_TM_score = f.readlines()
        
     
     clean_prediction = CLEAN_pred(iteration_num, ec_label)
     
     sequences_rep = dict()
     
     for line in rep_seq:
            if ">" in line:
                name = line.split("\t")[0].replace(">", "").replace("\n", "")
                emb_identifier = line.replace(">", "").replace("\n", "")
            else:
                aa = line.strip()
                sequences_rep[name] = {
                              "sequence" : aa,
                              "emb_identifier" : emb_identifier
                                      }
     
     
     # Build clean model for inference                                 
     model_clean = build_clean_model()

     # add the plddt
     sequences_rep = analyse_pdb(sequences_rep, iteration_num)

     # add activity
     sequences_rep = retieve_activity_prediction(sequences_rep, iteration_num, ec_label)

     for entry in alpha_TM_score:
            name = entry.split("\t")[0]
            TM = entry.split("\t")[2]
            TM_norm_que = entry.split("\t")[4]
            
            try: 
                clean_pred = clean_prediction[name]
                print(f'reward of seq {name} from clean: {clean_pred}')
            except:
                clean_pred = 0
                print(f'reward of seq {name} from clean: {clean_pred}-- ERROR')
            algn = int(entry.split("\t")[5])
            sequence = sequences_rep[str(name)]['sequence']
            lenght_rew = math.exp(-((((algn/len(sequence))-1)**2)/(0.5**2))) # Gaussian center on 1. The closer the ratio between len and aligment is, the higher is the reward
            emb_identifier = sequences_rep[str(name)]['emb_identifier']
            plddt = sequences_rep[str(name)]['plddt']
            activity = sequences_rep[str(name)]['activity']
            seq_emb = torch.load(f"./CLEAN/app/data/esm_data/{emb_identifier}.pt", weights_only=True)
            seq_emb = model_clean(seq_emb['mean_representations'][33].to(device))
            
            reference_emb = map_emb_center(ec_label)
         
            # Save in a dataset
            data["sequence"].append(formatting_sequence(sequence, ec_label))
            data["seq_name"].append(name)
            data["TM"].append(float(TM))
            data["TM_norm_que"].append(float(TM_norm_que))
            data["weight"].append((float(cosine_similarity(seq_emb, reference_emb)))*plddt*(lenght_rew))
     
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
     
     del model_clean
     return final_dataset
     
def CLEAN_pred(iteration_num, ec_label):
    target = ec_label
    with open(f'seq_gen_{ec_label}_iteration{iteration_num-1}_maxsep.csv', 'r') as f:
        data = f.read().split('\n')
    
    output = dict()

    for entry in data:
        if entry.strip():
            name = entry.split(',')[0]
            ec_list = entry.split(',')[1:]
            rewards = []
            current_dist = []
            reward = []
            
            for ec_entry in ec_list:
                    count = 0
                    current_EC = ec_entry.strip().split('/')[0][3:] 
                    current_dist.append(float(ec_entry.split('/')[1]))  
                    
                    for x in range(len(current_EC)):
                        try:
                            if target[:x] == current_EC[:x]:
                                
                                count += 1
                                
                        except:
                            pass
                    reward.append(count)
                    
            output[name.split('\t')[0]] = (statistics.mean(reward)-1)+(-statistics.mean(current_dist)) # as the distance can be zero, we add to the negative distance, maybe *10?
            
    return output

def build_clean_model():
    '''
    
    Load Clean Model and get the embeddings for each sequence to 
    compute cosine_similarity
    
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    model = LayerNormNet(512, 128, device, dtype)
    checkpoint = torch.load('./CLEAN/app/data/pretrained/split100.pth', weights_only=True, map_location=device) 
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def cosine_similarity(emb, reference_emb):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity = cos(emb, reference_emb)
    return similarity

def map_emb_center(target):
    '''
    Pluck out tensors fo the train set, and calculate the center of the specific ec cluster
    to calculate cos_similarity  
    '''

    with open('ec_lables_clean_list.txt', 'r') as f:
        ec_labels = f.read().split(',')

    train_emb = torch.load("./CLEAN/app/data/pretrained/100.pt", weights_only=True)
    try:
        positions = [x for x,y in enumerate(ec_labels) if y == target]
        reference_emb = train_emb[positions]
    except:
        print('EC label not valid') # might raise due to a typo or ec not present in the train set
    return reference_emb.mean(0)


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def formatting_sequence(sequence, ec_label):
    sequence = str(f"{ec_label}<sep><start>{sequence}<|endoftext|>")
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
  
  if int(iteration_num) == 1:

    model_name = "AI4PD/ZymCTRL"
    
  else:
    model_name = f"output_iteration{iteration_num-1}"
  
  print(f'Model {model_name} has been loaded')

  tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  ref_model = AutoModelForCausalLM.from_pretrained('AI4PD').to(device)
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

