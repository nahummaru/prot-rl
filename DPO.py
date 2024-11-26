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


# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.01,
    "seed": 1998,
    "learning_rate": 1e-7,
    "batch_size": 5,
    "num_epochs": 5,
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


def format_sequence(sequence, ec_label):
    """
    Formats a sequence for model input with special tokens.
    """
    return f"{ec_label}<sep><start>{sequence}<end><|endoftext|>"


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, ec_label):
    """
    Generates and preprocesses dataset for training and evaluation.
    """
    data = {"sequence": [], "seq_name": [], "TM": [], "TM_norm_que": [], "weight": []}

    with open(f"seq_gen_{ec_label}_iteration{iteration_num - 1}.fasta", "r") as f:
        rep_seq = f.readlines()

    with open(f"{ec_label}_TM_iteration{iteration_num - 1}", "r") as f:
        alpha_TM_score = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}

    for name, details in sequences_rep.items():
        sequence = details["sequence"]
        length_reward = math.exp(-(((len(sequence) / 237) - 1) ** 2) / (0.5**2))

        # Save in a dataset
        data["sequence"].append(format_sequence(sequence, ec_label))
        data["seq_name"].append(name)
        data["TM"].append(float(alpha_TM_score[0]))
        data["TM_norm_que"].append(float(alpha_TM_score[1]))
        data["weight"].append(length_reward)

    hf_dataset = Dataset.from_pandas(pd.DataFrame(data))
    shuffled_dataset = hf_dataset.shuffle(seed=CONFIG["seed"])

    train_size = int((1 - CONFIG["split_percent"]) * len(shuffled_dataset))
    train_dataset = shuffled_dataset.select(range(train_size))
    eval_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))

    final_dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
    final_dataset.save_to_disk(f"dataset_iteration{iteration_num}")

    return final_dataset


# ---------------------------
# Model Functions
# ---------------------------
def log_likelihood(sequences, device, model, tokenizer):
    """
    Calculates log likelihood for a batch of sequences.
    """
    all_loss = []
    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors="pt").to(device)
        outputs = model(inputs, labels=inputs)
        loss, _ = outputs[:2]
        all_loss.append(loss.unsqueeze(0))
    return torch.cat(all_loss)


def dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta=0.1):
    """
    Calculates the weighted loss using Dynamic Policy Optimization (DPO).
    """
    log_ratios = beta * (policy_log_probs - ref_log_probs) if ref_log_probs is not None else beta * policy_log_probs
    weights = torch.softmax(weights * -1, dim=0)
    return F.cross_entropy(log_ratios, weights)


# ---------------------------
# Training and Evaluation
# ---------------------------
def train(model, ref_model, tokenizer, train_loader, optimizer, device):
    """
    Performs training for one epoch.
    """
    model.train()
    total_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        sequences = batch["sequence"]
        ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
        policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
        weights = batch["weight"].to(device)

        loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


def evaluate(model, ref_model, tokenizer, eval_loader, device):
    """
    Evaluates the model on the evaluation set.
    """
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in eval_loader:
            sequences = batch["sequence"]
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
            weights = batch["weight"].to(device)

            loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


# ---------------------------
# Main Function
# ---------------------------
def main(train_loader, eval_loader, iteration_num, model_directory):
    """
    Main training loop for a given iteration.
    """
    model_name = model_directory if iteration_num == 1 else f"output_iteration{iteration_num - 1}"
    print(f"Model {model_name} has been loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained("path_to_reference_model").to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["adam_decay"],
    )

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train(model, ref_model, tokenizer, train_loader, optimizer, device)
        eval_loss = evaluate(model, ref_model, tokenizer, eval_loader, device)
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        save_model_and_tokenizer(model, tokenizer, output_dir=f"output_iteration{iteration_num}")

    del model, ref_model


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    args = parser.parse_args()
    seed_everything(CONFIG["seed"])

    if not os.path.exists(f"dataset_iteration{args.iteration_num}"):
        dataset = generate_dataset(args.iteration_num, args.label.strip())
    else:
        dataset = load_from_disk(f"dataset_iteration{args.iteration_num}")

    print("Dataset Loaded!")
    train_loader = DataLoader(dataset["train"], batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(dataset["eval"], batch_size=CONFIG["batch_size"], shuffle=False)

    main(train_loader, eval_loader, args.iteration_num, args.model_dir)
