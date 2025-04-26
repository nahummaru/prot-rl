import torch, os, math, argparse
from transformers import GPT2LMHeadModel, AutoTokenizer

def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids, model, tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model, special_tokens, device, tokenizer):
    
    # Generating sequences
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids, 
        top_k=9,
        repetition_penalty=1.2,
        max_length=1024,
        eos_token_id=1,
        pad_token_id=0,
           do_sample=True,
           num_return_sequences=20) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    # Check sequence sanity, ensure sequences are not-truncated.
    # The model will truncate sequences longer than the specified max_length (1024 above). We want to avoid those sequences.
    new_outputs = [ output for output in outputs if output[-1] == 0]
    if not new_outputs:
        print("not enough sequences with short lengths!!")

    # Compute perplexity for every generated sequence in the batch
    ppls = [(tokenizer.decode(output), calculatePerplexity(output, model, tokenizer)) for output in new_outputs ]

    # Sort the batch by perplexity, the lower the better
    ppls.sort(key=lambda i:i[1]) 

    # Final dictionary with the results
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, default=1)
    parser.add_argument("--label", type=str)
    parser.add_argument("--cwd", type=str)
    
    args = parser.parse_args()
    iteration_num = int(args.iteration_num)
    label = args.label
    cwd = args.cwd

    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    if iteration_num == 0: 
        tokenizer = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL") # change to ZymCTRL location
        model = GPT2LMHeadModel.from_pretrained("AI4PD/ZymCTRL").to(device) # change to ZymCTRL location
    else: 
        checkpoint = [x for x in os.listdir(f'{cwd}models/{label}_model{iteration_num}') if 'checkpoint' in x][0] ## changed 8/10/2024
        tokenizer = AutoTokenizer.from_pretrained(f"{cwd}models/{label}_model{iteration_num}/{checkpoint}") # change to ZymCTRL location
        model = GPT2LMHeadModel.from_pretrained(f"{cwd}models/{label}_model{iteration_num}/{checkpoint}").to(device) # change to ZymCTRL location

    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    out = str()
    error_seqs = str()
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY\n")
    # We'll run 100 batches per label. 20 sequences will be generated per batch.
    for i in range(0,100):
        sequences = main(label, model, special_tokens, device, tokenizer)
        for key,value in sequences.items():
            for index, val in enumerate(value):
                if all(char in canonical_amino_acids for char in val[0]):
                    out = out + (f'>{label}_{i}_{index}_iteration{iteration_num}\t{val[1]}\n{val[0]}') + ('\n')
                    
 # Sequences will be saved with the name of the label followed by the batch index,
 # and the order of the sequence in that batch.           
    fn = open(f"{cwd}generated_sequences/seq_gen_{label}_iteration{iteration_num}.fasta", "w")
    fn.write(str(out))
    fn.close()

    del model, tokenizer, out, sequences