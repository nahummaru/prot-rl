import json
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from tqdm import tqdm
import math
import argparse



def prepare_af3_json(name, seq, iteration_num):

    directory_name = f"alphafold3_input_iteration_{iteration_num}"
    os.makedirs(directory_name, exist_ok=True)
    with open('json_template_CD80.json', 'r') as file:
      json_template = json.load(file)
    
    json_template["name"] = name
    json_template["sequences"][0]["protein"]["sequence"] = seq
    
    doc_name = f"{directory_name}/{name}.json"

    with open(doc_name, "w") as f:
        json.dump(json_template, f, indent=2)


def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq



def remove_char(sequence):
    return sequence.replace("<|endoftext|>","").replace("\n","")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    label = args.label
    labels = [args.label]

    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')

    model_name = f"output_iteration{iteration_num}"

    if iteration_num == 0:
      model_name = args.model_dir
  
    print(f'Model {model_name} has been loaded')

    protgpt2 = pipeline('text-generation', model=model_name)
    index = 0
    all_sequences = []
    for i in range (20):
      sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)
      for seq in sequences:
        sequence_id = f"{label}_{i}_{index}_iteration{iteration_num}"
        cleaned_seq = remove_char(seq["generated_text"])
        
        all_sequences.append(f">{sequence_id}\n{cleaned_seq}\n")
        #print(f">{sequence_id}\n{cleaned_seq}")
        
        prepare_af3_json(sequence_id, cleaned_seq, iteration_num)
        
        index += 1

    fasta_content = ''.join(seq for seq in all_sequences)
          
    output_filename = f"seq_gen_{label}_iteration{iteration_num}.fasta"
    with open(output_filename, "w") as fn:
          fn.write(fasta_content)

