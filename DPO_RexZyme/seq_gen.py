from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
import torch
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--label", type=str)

args = parser.parse_args()
iteration_num = args.iteration_num
moodel_dir =  args.model_dir
label = args.label.strip()

device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    
tokenizer_aa = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/REXzyme_aa")
print("tokenizer_aa loaded")

tokenizer_smiles = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/REXzyme_smiles")
print("tokenizer_smiles_lodaded")

if iteration_num == 0:
      model_name = moodel_dir
else:
      model_name = f'output_iteration{iteration_num}'
    

model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
print(f"model {model_name} loaded")
#print(model.generation_config)


# Reactions are in a file separated by end of lime and between '
# example: ['[Fe+2].[Fe+2].[Fe+2].[Fe+2].[H+].[H+].[H+].[H+].[H+].[H+].[H+].[H+].O=O>>[Fe+3].[Fe+3].[Fe+3].[Fe+3].[H+].[H+].[H+].[H+].[H]O[H].[H]O[H]'

reactions=["O=C([O-])O.[H+]>>O.O=C=O"]

def calculatePerplexity(inputs,model):
    '''Function to compute perplexity'''
    a=tokenizer_aa.decode(inputs)
    b=tokenizer_aa(a, return_tensors="pt").input_ids.to(device='cuda')
    b = torch.stack([[b[b!=tokenizer_aa.pad_token_id]] for label in b][0])
    with torch.no_grad():
        outputs = model(b, labels=b)
    loss, logits = outputs[:2]
    return math.exp(loss)


for idx, i in tqdm(enumerate(reactions)):
    print(f"{i}")
    out = ""
    input_ids = tokenizer_smiles(f"translation{i}", return_tensors="pt").input_ids.to(device='cuda')
    print(f'Generating for {input_ids}')
    ppls_total = []
    for _ in range(1):
        outputs = model.generate(input_ids,
                                  top_k=18,
                                  top_p=1,
                                  repetition_penalty=1.1,
                                  max_length=1024,
                                  do_sample=True,
                                  eos_token_id = 1,
                                  num_return_sequences=200)
        ppls = [(tokenizer_aa.decode(output, skip_special_tokens=True), calculatePerplexity(output, model), len(tokenizer_aa.decode(output, skip_special_tokens=False))) for output in tqdm(outputs)]
        ppls_total.extend(ppls)
        ppls_total.sort(key=lambda i:i[1])
    with open(f'FDH_reac_generated_iteration{iteration_num}.fasta', 'w') as fn:
        for ix, j in enumerate(ppls_total):
            fn.write(f">{ix}_{j[1]}_{j[2]}\n{j[0]}\n")