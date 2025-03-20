import os
import math
import random
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import json
import tempfile
import glob
from glob import glob
import csv
from collections import defaultdict

from functions import *

import pdbfixer
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import io



# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.1,
    "seed": 1998,
    "learning_rate": 1e-5,
    "batch_size": 20,
    "num_epochs": 10,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Utility Functions
# ---------------------------
def seed_everything(seed=2003):
    """
    Sets random seed for reproducibility across libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def append_to_csv(
    name,
    pMPNN,
    d_rmsd,
    pAE_bt,
    shape_complimentary,
    uns_hydrogens,
    hydrophobicity,
    binder_score,
    interface_dSASA,
    sequence,
    iteration_num,
    score,
    plddt,
    i_pTM,
    pAE_b,
    ipsae,
    helicity,
    lenght,
    pae,
    ptm,
    has_clash,
    pAE_t,
    pae_2,
    contact_probs,
    output_file
):
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "pMPNN",
            "d_rmsd",
            "pAE_bt",
            "shape_complimentary",
            "uns_hydrogens",
            "hydrophobicity",
            "binder_score",
            "interface_dSASA",
            "sequence",
            "iteration_num",
            "score",
            "plddt",
            "i_pTM",
            "pAE_b",
            "ipsae",
            "helicity",
            "lenght",
            "ptm",
            "has_clash",
            "pAE_t",
            "pae_2",
            "contact_probs",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "pMPNN": pMPNN,
            "d_rmsd": d_rmsd,
            "pAE_bt": pAE_bt,
            "shape_complimentary": shape_complimentary,
            "uns_hydrogens": uns_hydrogens,
            "hydrophobicity": hydrophobicity,
            "binder_score": binder_score,
            "interface_dSASA": interface_dSASA,
            "sequence": sequence,
            "iteration_num": iteration_num,
            "score": score,
            "plddt": plddt,
            "i_pTM": i_pTM,
            "pAE_b": pAE_b,
            "ipsae": ipsae,
            "helicity": helicity,
            "lenght": lenght,
            "ptm": ptm,
            "has_clash": has_clash,
            "pAE_t": pAE_t,
            "pae_2": pae_2,
            "contact_probs": contact_probs
        })


def models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True


def formatting_sequence(sequence):
    """
    Formats correctly the sequence as in the ProtGPT2 trainset.
    """
    return "<|endoftext|>"+"\n".join(sequence[i:i+60] for i in range(0, len(sequence), 60))+"<|endoftext|>"


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def af_metrics(name, path):
    name = name.lower()
    metrics_file = f"{name}_summary_confidences.json"
   
    with open(os.path.join(path, metrics_file), "r") as f:
        metrics_summary = json.load(f)

    with open(os.path.join(path, metrics_file.replace("summary_","")), "r") as f:
        metrics = json.load(f)

    pae = np.array(metrics["pae"])
    ptm = metrics_summary['ptm']
    iptm = metrics_summary['iptm']
    has_clash = metrics_summary['has_clash']
    
    #ipsae = get_ipsae(path, arg1=10, arg2=10)
    ipsae = 1
    
    chain_ids = metrics["token_chain_ids"]  
    atom_ids = metrics["atom_chain_ids"]
    plddt = metrics['atom_plddts']

    chain_ids_binder = [x for x in chain_ids if x == "B"]
    atom_ids_binder = [x for x in atom_ids if x == "B"]
    
    plddt = np.array(plddt[:len(atom_ids_binder)]).mean()

    pae = np.array(metrics["pae"])
    b_pae = pae[len(chain_ids_binder):, :len(chain_ids_binder)].mean()
    t_pae = pae[:len(chain_ids_binder), len(chain_ids_binder):].mean()

    pae_2 = (b_pae.mean() + t_pae.mean()) / 2

    return iptm, pae , plddt, ptm, has_clash, ipsae, b_pae, t_pae, pae_2

def get_chain_indices(chain_ids):

    chain_map = defaultdict(list)
    for i, c in enumerate(chain_ids):
        chain_map[c].append(i)
    return dict(chain_map)


def compute_inter_chain_contacts(contact_probs, chain_map, chain1='A', chain2='B'):

    idx_chain1 = chain_map.get(chain1, [])
    idx_chain2 = chain_map.get(chain2, [])
    
    if not idx_chain1 or not idx_chain2:
        raise ValueError(f"No residues found for chain {chain1} or chain {chain2}!")
    
    # Convert to numpy arrays (just for safety)
    idx_chain1 = np.array(idx_chain1)
    idx_chain2 = np.array(idx_chain2)
    
    # Extract the submatrix of probabilities for chain1 vs chain2
    sub_probs = contact_probs[np.ix_(idx_chain1, idx_chain2)]

    return sub_probs


def load_alphafold_data(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_contact_points(sub_probs, chain_ids):
    contact = []
    for i in range(len(chain_ids)):
        contact.append(sum(sub_probs[:,i]))
        
    return np.array(contact)

def ratio_contacted_key_residues(name, hotspot_residues, interface_residues_pdb_ids_target_str):

    interface_residues_list = interface_residues_pdb_ids_target_str.replace("A", "").split(",")
    interface_residues_list = [int(x) for x in interface_residues_list if x.strip()]  # include a check for empty strings
    hotspot_residues = [int(x) for x in hotspot_residues]
    common_elements = set(hotspot_residues) & set(interface_residues_list)
    
    contact_prob = len(common_elements) / len(hotspot_residues)
    print(len(common_elements), len(hotspot_residues))
    
    return contact_prob

def get_pMPNN(pdb_file):
    

    with tempfile.TemporaryDirectory() as output_dir:
       
            command_line_arguments = [
                "python",
                "/home/woody/b114cb/b114cb23/ProteinMPNN/protein_mpnn_run.py",
                "--pdb_path", pdb_file,
                "--pdb_path_chains", "B",
                "--score_only", "1",
                "--save_score", "1",
                "--out_folder", output_dir,
                "--batch_size", "1"
            ]

            proc = subprocess.run(command_line_arguments, stdout=subprocess.PIPE, check=True)
            output = proc.stdout.decode('utf-8')
            for x in output.split('\n'):
                if x.startswith('Score for'):
                                name = x.split(',')[0][10:-9]
                                mean =x.split(',')[1].split(':')[1]
    return float(mean)

def convert_cif_to_pdb(cif_file):
    """
    Converts a CIF file to PDB format and returns the PDB string.
    """
    fixer = PDBFixer(cif_file)

    # Handle missing atoms/residues if needed
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()

    # Store PDB data in a string buffer
    pdb_file = cif_file.replace("cif","pdb")
    with open(pdb_file, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

def get_ipsae(json_path, arg1=10, arg2=10):
    # run ipsae.py
    command = ["python", "./functions/ipsae.py", json_path, str(arg1), str(arg2)]
    subprocess.run(command)
    output_path=""
    with open(output_path, "r") as f:
        ipsae_data = f.read()
    ipsae = 1 
    #Process data

    return ipsae

def calc_ss_percentage(pdb_file):
    # Parse the structure

    chain_id="B"
    atom_distance_cutoff=4.0

    with open("/home/woody/b114cb/b114cb23/ProtWrap/functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues, _ = hotspot_residues(pdb_file, chain_id, atom_distance_cutoff)
    interacting_residues = set(interacting_residues.keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def py_ros_score_interface(pdb_file):
    
    with open("/home/woody/b114cb/b114cb23/ProtWrap/functions/default_4stage_multimer.json", "r") as f:
        advanced_settings = json.load(f)

    pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball "/home/woody/b114cb/b114cb23/ProtWrap/functions/DAlphaBall.gcc" -corrections::beta_nov16 true -relax:default_repeats 1')
    
    print(f"Scoring interface of {pdb_file}")
    interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str = score_interface(pdb_file, binder_chain="B")
    print(f"Target interface residues: {interface_residues_pdb_ids_target_str}")
    return interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str
 

# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, label, mode):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }
    seq_lenght = []
    with open(f"seq_gen_{label}_iteration{iteration_num-1}.fasta", "r") as f:
        rep_seq = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}
    
    
    hotspot_residues = [18,43,44,46,49,50,53,61,62,64,65,66,68,69,70,72,73,75,76,77,80]

    for entry in sequences_rep:
        try:
            name = entry
            sequence = sequences_rep[str(name)]['sequence']
            lenght = math.exp(-((((90/len(sequence))-1)**2)/(0.5**2))) # Gaussian center on 1. The closer the ratio between len and aligment is, the higher is the reward
            
            path = f"./alphafold_output_iteration{iteration_num-1}/{name.lower()}"
            i_pTM, pae , plddt, ptm, has_clash, ipsae, pAE_b, pAE_t, pae_2 = af_metrics(name.lower(), path)
            
            pAE_bt = (pAE_b + pAE_t)/2

            cif_file = path + f"/{name.lower()}_model.cif"
            convert_cif_to_pdb(cif_file)
            pdb_file = path + f"/{name.lower()}_model.pdb"
            
            interface_scores, interface_AA, interface_residues_pdb_ids_str, interface_residues_pdb_ids_target_str = py_ros_score_interface(pdb_file)
            
            print(interface_residues_pdb_ids_target_str)

            helicity, trajectory_beta, trajectory_loops, trajectory_alpha_interface, trajectory_beta_interface, trajectory_loops_interface, i_plddt, trajectory_ss_plddt = calc_ss_percentage(pdb_file)
            
            pMPNN = get_pMPNN(pdb_file)
            contact_probs = ratio_contacted_key_residues(name, hotspot_residues, interface_residues_pdb_ids_target_str)
            print(interface_residues_pdb_ids_target_str)
            print(f"Contact probs: {contact_probs}")
            shape_complimentary = interface_scores["interface_sc"]
            uns_hydrogens = interface_scores["interface_delta_unsat_hbonds"]
            hydrophobicity = interface_scores["surface_hydrophobicity"]
            binder_score = interface_scores["binder_score"]
            interface_dSASA = interface_scores["interface_dSASA"]
            d_rmsd = target_pdb_rmsd(pdb_file, "/home/woody/b114cb/b114cb23/ProtWrap/4rwh.pdb", "A")
            print(f"Sequence: {name}, plddt: {plddt}, i_pTM: {i_pTM}, pAE_b: {pAE_b}, ipsae: {ipsae}, pAE_bt: {pAE_bt}, helicity: {helicity}, pMPNN: {pMPNN}, lenght: {lenght}")

            score = 0.1*plddt + 0.5*i_pTM - 0.04*pAE_b + 0.4*ipsae + 0.01*pAE_bt + (- 0.03)*helicity - pMPNN + 10*lenght + 0.03*shape_complimentary + 0.1*hydrophobicity + 0.003*binder_score + 0.003*interface_dSASA - 0.4*uns_hydrogens + 5*contact_probs + 0.5*d_rmsd
            
            append_to_csv(name, -pMPNN, d_rmsd, pAE_bt, shape_complimentary, uns_hydrogens, hydrophobicity,
                            binder_score, interface_dSASA, sequence, iteration_num, score, plddt, i_pTM,
                            pAE_b, ipsae, helicity, lenght, pae, ptm, has_clash, pAE_t, pae_2,contact_probs, "logs_output.cvs")
            
            data["sequence"].append(formatting_sequence(sequence))
            data["seq_name"].append(entry)
            data["weight"].append(score)
        except:
                print(f"error for sequence {name}")
    # Convert data dictionary to a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Prepare pairs if mode is 'paired'
    if mode == 'paired':
        hf_dataset = prepare_pairs(hf_dataset)

    # Shuffle and split the dataset
    shuffled_dataset = hf_dataset.shuffle(seed=CONFIG["seed"])
    train_size = int((1 - CONFIG["split_percent"]) * len(shuffled_dataset))
    train_dataset = shuffled_dataset.select(range(train_size))
    eval_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))

    # Save the dataset to disk and return
    final_dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
    final_dataset.save_to_disk(f"dataset_iteration{iteration_num}")

    return final_dataset


def prepare_pairs(hf_dataset):
    """
    Prepare paired data from the paired form of DPO.
    """
    # Sort the dataset by weight in descending order
    sorted_dataset = hf_dataset.sort("weight", reverse=False)

    # Split the dataset into two halves
    mid_point = len(sorted_dataset) // 2
    first_half = sorted_dataset.select(range(mid_point))
    second_half = sorted_dataset.select(range(mid_point, len(sorted_dataset)))

    # Create pairs of positive and negative sequences
    pairs = []
    for pos_example, neg_example in zip(first_half, second_half):
        pairs.append({
            "positive_sequence": pos_example["sequence"],
            "negative_sequence": neg_example["sequence"],
        })

    return Dataset.from_list(pairs)


# ---------------------------
# Loss Functions
# ---------------------------
def log_likelihood(sequences, device, model, tokenizer):
    
    all_log_likelihood = []  # List to store loss for each sequence

    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
        outputs = model(inputs, labels=inputs)
        neg_log_likelihood, logits = outputs[:2]                        # The HF loss output is the negative log-likelihood averaged over the number of tokens.
        all_log_likelihood.append(-neg_log_likelihood.unsqueeze(0)) # Convert negative log-likelihood to likelihood by multiplying by -1.
        
    all_log_likelihood = torch.cat(all_log_likelihood)
    
    return all_log_likelihood

def dpo_paired_loss(batch, model, ref_model, tokenizer, device, beta=0.1):
    """
    Calculates the paired DPO loss.
    """
    # Extract positive and negative sequences
    positive_sequence = batch["positive_sequence"]
    negative_sequence = batch["negative_sequence"]

    # Log probabilities for positive sequences
    pos_ref_log_probs = log_likelihood(positive_sequence, device, ref_model, tokenizer)
    pos_policy_log_probs = log_likelihood(positive_sequence, device, model, tokenizer)
    if models_equal(ref_model, model):
                pos_ref_log_probs = None
                
    if pos_ref_log_probs is None:
        pos_ratios = beta * pos_policy_log_probs
    else:
        pos_ratios = beta * (pos_policy_log_probs - pos_ref_log_probs)

    # Log probabilities for negative sequences
    neg_ref_log_probs = log_likelihood(negative_sequence, device, ref_model, tokenizer)
    neg_policy_log_probs = log_likelihood(negative_sequence, device, model, tokenizer)
    if neg_ref_log_probs is None:
        neg_ratios = beta * (neg_policy_log_probs)
    else:
        neg_ratios = beta * (neg_policy_log_probs - neg_ref_log_probs)

    # Compute the DPO paired loss
    loss = -F.logsigmoid(pos_ratios - neg_ratios)

    return  torch.mean(loss)
    
def dpo_weighted_loss(pi_log_likelihood, ref_log_likelihood, weights, beta=0.1):
    """
    Calculates DPO weighted loss. 
    Function kindly provided by Widatalla et.al 2024 "Aligning protein 
    generative models with experimental fitness via Direct Preference Optimization"
    """
    if ref_log_likelihood is None:
        pi_ratio = beta * pi_log_likelihood
    else:
        pi_ratio = beta * (pi_log_likelihood - ref_log_likelihood)
        
    weights = torch.softmax(weights, dim=0)
    loss = F.cross_entropy(pi_ratio, weights)
    
    return loss


def dpo_ranked_loss(pi_log_likelihood, pi_ref_loglikelihood, weights, beta=0.1):
    """
    Calculates the Directed Policy Optimization (DPO) ranked loss.
    In this case the ranking is on the batch dimension.
    """
    # Ensure weights have at least one dimension
    weights = torch.softmax(weights, dim=0)
    weights = weights.view(-1)  
    
    sorted_indices = torch.argsort(weights, descending=True)
    pi_log_likelihood = pi_log_likelihood[sorted_indices]
    pi_ref_loglikelihood = pi_ref_loglikelihood[sorted_indices] if not pi_ref_loglikelihood == pi_log_likelihood else None
    weights = weights[sorted_indices]
    print(f"Sorted weights: {weights}")

    if pi_ref_loglikelihood is None:
        pi_ratio = beta * pi_log_likelihood
    else:
        pi_ratio = beta * (pi_log_likelihood - pi_ref_loglikelihood)

    uniform_weights = torch.ones_like(pi_ratio)
    print(f"pi ratios: {pi_ratio}")

    
    loss = F.cross_entropy(pi_ratio, uniform_weights)
    return loss



# ---------------------------
# Training and Evaluation
# ---------------------------
import copy
import torch

def train(model, iteration_num, ref_model, tokenizer, train_loader, optimizer, device, mode):
  
    model.train()
    total_loss = []
    for batch in train_loader:
        # Save a snapshot of parameters before this update
        prev_state = copy.deepcopy(model.state_dict())

        if mode != 'paired':
            optimizer.zero_grad()
            sequences = batch["sequence"] 
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences, device, model, tokenizer)

            if iteration_num == 1:
                ref_log_probs = None

            weights = batch["weight"].to(device)

            if mode == "weighted":
                loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            elif mode == "ranked":
                loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            
        elif mode == "paired":
            loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])
        
        loss.backward()
        optimizer.step()

        # Compare current state to previous snapshot
        new_state = model.state_dict()
        for key in prev_state:
            if not torch.allclose(prev_state[key], new_state[key]):
                print(f"Parameter '{key}' has changed after update.")
            else:
                print(f"Parameter '{key}' is unchanged after update.")

        total_loss.append(loss.item())
    
    torch.cuda.empty_cache()
    return sum(total_loss) / len(total_loss)



def evaluate(model, iteration_num, ref_model, tokenizer, eval_loader, optimizer, device, mode):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_loader:
            if mode != 'paired':
                sequences = batch["sequence"]
                ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
                policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
                
                if models_equal(ref_model, model) or iteration_num == 1:
                    ref_log_probs = None
                
                weights = batch["weight"].to(device)

                if mode == "weighted":
                    loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])

                elif mode == "ranked":
                    loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])

            else:
                # Paired mode
                loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])

            total_loss += loss.item()

    # Take the average loss across all batches in the eval_loader
    return total_loss / len(eval_loader)


# ---------------------------
# Main Function
# ---------------------------
def main(train_loader, eval_loader, iteration_num, model_directory, mode):
    """
    Main training loop for a given iteration.
    """
    model_name = model_directory if iteration_num == 1 else f"output_iteration{iteration_num - 1}"
    print(f"Model {model_name} has been loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_directory).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["adam_decay"],
    )

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train(model, iteration_num, ref_model, tokenizer, train_loader, optimizer, device, mode)
        eval_loss = evaluate(model, iteration_num, ref_model, tokenizer, eval_loader, optimizer, device, mode)
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        save_model_and_tokenizer(model, tokenizer, output_dir=f"output_iteration{iteration_num}")

    del model
    del ref_model
    torch.cuda.empty_cache()

# ---------------------------
#     MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()
    seed_everything(CONFIG["seed"])

    if not os.path.exists(f"dataset_iteration{args.iteration_num}"):
        dataset = generate_dataset(args.iteration_num, args.label.strip(), args.mode)
    else:
        dataset = load_from_disk(f"dataset_iteration{args.iteration_num}")

    print("Dataset Loaded!")
    train_loader = DataLoader(dataset["train"], batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(dataset["eval"], batch_size=CONFIG["batch_size"], shuffle=False)

    main(train_loader, eval_loader, args.iteration_num, args.model_dir, args.mode)
