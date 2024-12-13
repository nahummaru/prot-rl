import torch, os, math, argparse
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM

def remove_characters(sequence, char_list):
    '''
    Removes special tokens used during training.
    '''
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids, model, tokenizer):
    '''
    Computes perplexities for the generated sequences. 
    '''
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def main(label, model, special_tokens, device, tokenizer):
    '''
    Function to generate sequences from the loaded model.
    '''
    input_ids = tokenizer.encode(label,return_tensors='pt').to(device)
    # Generating sequences
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

    # Final results dictionary without strange characters
    sequences={}
    sequences[label] = [(remove_characters(x[0], special_tokens), x[1]) for x in ppls]

    return sequences

if __name__=='__main__':

    # parse the arguments of the script 
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--ec_label", type=str)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.ec_label

    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    
    if iteration_num == 0:
      model_name = 'AI4PD/ZymCTRL' # will load ZymCTRL using the transformers' API 
    else:
      model_name = f'output_iteration{iteration_num}'
    
    print(f'{model_name} loaded in {device}')
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    out = str()
    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # Set of canonical amino acids
    
    # To not saturate GPU memory, we'll generate the sequences in batches of 20, to change the amount modify n_batches or the num_return_sequences in main()
    n_batches = 5 # this will generate 5 batches of 20 sequence: 100 sequences in total (all saved into the same fasta)
    for i in range(n_batches):
        # Generate the batch of sequences 
        sequences = main(ec_label, model, special_tokens, device, tokenizer)
        for key, value in sequences.items():
            for index, val in enumerate(value):
                if all(char in canonical_amino_acids for char in val[0]): # remove sequences with characters out of the 20 canonical amino acids 
                    out = out + (f'>{ec_label}_{i}_{index}_iteration{iteration_num}\t{val[1]}\n{val[0]}') + ('\n') # store the generated sequences into a string 
    
    output_filename = f"seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    print(f'Generated sequences saved in {output_filename}')
    with open(output_filename, "w") as fn:
        fn.write(out)

    # free memory resources 
    del model 
    torch.cuda.empty_cache()

    
