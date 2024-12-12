import os, torch
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse

torch.cuda.empty_cache()

##### Load the module ESM ######
tokenizer_esm = AutoTokenizer.from_pretrained("facebook/esmfold_v1") # Download tokenizer
model_esm = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")  # Download model
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model_esm = model_esm.to(device)
print('The ESM model is on CUDA: ', next(model_esm.parameters()).is_cuda)

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, default=1)
parser.add_argument("--label", type=str)
parser.add_argument("--cwd", type=str)

args = parser.parse_args()
iteration_num = args.iteration_num
label = args.label
cwd = args.cwd

model_esm.eval()

# Put sequences into dictionary
with open(f"{cwd}generated_sequences/seq_gen_{label}_iteration{iteration_num}.fasta", "r") as f:
    data = f.readlines()


sequences={}
for line in data:
    if '>' in line:
        name = line.strip()
        sequences[name] = str()  #! CHANGE TO corre
        continue
    sequences[name] = line.strip()

count = 0
error = 0

for name, sequence in sequences.items():
  name = name[1:]
  name = name.split("\t")[0]
  if os.path.exists(f'{cwd}PDB/{label}_output_iteration{iteration_num}'): 
    already = list(os.listdir(f'{cwd}PDB/{label}_output_iteration{iteration_num}'))
  else: 
     already = []
  if name+'.pdb' not in already:
    try:
      count += 1  
      with torch.no_grad():
        output = model_esm.infer_pdb(sequence)
        torch.cuda.empty_cache()
        os.makedirs(f"{cwd}PDB/{label}_output_iteration{iteration_num}", exist_ok=True)
        with open(f"{cwd}PDB/{label}_output_iteration{iteration_num}/{name}.pdb", "w") as f:
              f.write(output)
    except:
      error += 1
      
      print(f"Number of errors: {error}")
      torch.cuda.empty_cache()
del model_esm, tokenizer_esm