import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse
import os


##### Load the module ESM ######
tokenizer_esm = AutoTokenizer.from_pretrained("./esm_fold") # Download tokenizer
model_esm = EsmForProteinFolding.from_pretrained("./esm_fold")  # Download model
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model_esm = model_esm.to(device)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int)
parser.add_argument("--label", type=str)
args = parser.parse_args()
iteration_num = args.iteration_num
ec_label = args.label
ec_label = ec_label.strip()
    
    

model_esm.eval()


# Put sequences into dictionary
with open(f"seq_gen_{ec_label}_iteration{iteration_num}.fasta", "r") as f:
    data = f.readlines()


sequences={}
for line in data:
    if '>' in line:
        name = line.strip()
        sequences[name] = str()  #! CHANGE TO corre
        continue
    sequences[name] = line.strip()

print(len(sequences))


count = 0
error = 0

for name, sequence in sequences.items():
  try:
    count += 1  
    with torch.no_grad():
      output = model_esm.infer_pdb(sequence)
      torch.cuda.empty_cache()
      name = name[1:]
      name = name.split("\t")[0]
      os.makedirs(f"output_iteration{iteration_num}/PDB", exist_ok=True)
      with open(f"output_iteration{iteration_num}/PDB/{name}.pdb", "w") as f:
            f.write(output)
  except:
    error += 1
    
    #print(f'Sequence {name} is processed. {len(sequences)-count} remaining!') 
    print(f"Number of errors: {error}")
    torch.cuda.empty_cache()
del model_esm

