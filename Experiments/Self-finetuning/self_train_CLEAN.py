import random, transformers, subprocess, os, argparse, statistics, torch, math
from transformers import AutoTokenizer
import random
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoTokenizer
from transformers.testing_utils import CaptureLogger

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

def cosine_similarity(emb, reference_emb):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity = cos(emb, reference_emb)
    return similarity

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

def build_clean_model(CLEAN_app_path):
    '''
    
    Load Clean Model and get the embeddings for each sequence to 
    compute cosine_similarity
    
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    model = LayerNormNet(512, 128, device, dtype)
    checkpoint = torch.load(f'{CLEAN_app_path}data/pretrained/split100.pth', weights_only=True, map_location=device) 
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    # del checkpoint
    return model

def map_emb_center(target, cwd, CLEAN_app_path):
    '''
    Pluck out tensors fo the train set, and calculate the center of the specific ec cluster
    to calculate cos_similarity  
    '''

    with open(f'{cwd}scripts/ec_labels_clean_list.txt', 'r') as f:
        ec_labels = f.read().split(',')

    train_emb = torch.load(f"{CLEAN_app_path}data/pretrained/100.pt")
    try:
        positions = [x for x,y in enumerate(ec_labels) if y == target]
        reference_emb = train_emb[positions]
    except:
        print('EC label not valid') # might raise due to a typo or ec not present in the train set
    return reference_emb.mean(0)

def CLEAN_pred(clean_file, label):
    target = label
    with open(clean_file, 'r') as f:
        data = f.read().split('\n')
    
    output = dict()
    
    for entry in data:
        if entry.strip():
            name = entry.split(',')[0]
            ec_list = entry.split(',')[1:]
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

def generate_dataset(generated_sequences, clean_file, label, cwd, CLEAN_app_path):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }
    
    with open(generated_sequences, "r") as f:
        rep_seq = f.readlines()
        
    sequences_rep = dict()
     
    clean_prediction = CLEAN_pred(clean_file, label)

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
    model_clean = build_clean_model(CLEAN_app_path)

    for name, values in sequences_rep.items():
        if name in clean_prediction.keys():
            try: 
                clean_pred = clean_prediction[name]
                print(f'reward of seq {name} from clean: {clean_pred}')
            except:
                clean_pred = 0
                print(f'reward of seq {name} from clean: {clean_pred}-- ERROR')
            if clean_pred != 0: 
                sequence = sequences_rep[str(name)]['sequence']

                lenght_rew = math.exp(-((((len(sequence)/260)-1)**2)/(0.5**2))) # Gaussian center on 1. The closer the ratio between len and aligment is, the higher is the reward 
                emb_identifier = values['emb_identifier']
                
                seq_emb = torch.load(f"{CLEAN_app_path}/data/esm_data/{emb_identifier}.pt", weights_only=True)
                seq_emb = model_clean(seq_emb['mean_representations'][33].to(device))
                
                reference_emb = map_emb_center(label, cwd, CLEAN_app_path)

                data["sequence"].append(sequence)
                data["seq_name"].append(name)
                data["weight"].append((float(cosine_similarity(seq_emb, reference_emb)))*(lenght_rew)) # this is the reward function that scores the synthtic sequences 

    del model_clean
    return data

def grouper(iterable):
    '''
    Group the sequences for the finetuning step.
    '''
    prev = None
    group = ''
    total_sum = 0
    for item in iterable:
        if prev is None or item[0] + total_sum < 1025:
            group += item[1]
            total_sum += item[0]
        else:
            total_sum = item[0]
            yield group
            group = item[1]
        prev = item
    if group:
        total_sum = 0
        yield group

def tokenize_function(examples):
    '''
    '''
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples["text"])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above'- this long input will be chunked into smaller bits before being passed to the model."
        )
    return output

def group_texts(examples, block_size=1024):
    '''
    Concatenate all texts.
    '''
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop,
    # you can customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    parser.add_argument("--cwd", type=str)
    parser.add_argument("--CLEAN_app_path", type=str)

    args = parser.parse_args()
    iteration_num = int(args.iteration_num)
    label = str(args.label)
    cwd = args.cwd
    CLEAN_app_path = args.CLEAN_app_path

    generated_sequences = f'{cwd}generated_sequences/seq_gen_{label}_iteration{iteration_num-1}.fasta'
    clean_file = f'{cwd}clean/seq_gen_{label}_iteration{iteration_num-1}_maxsep.csv'

    data = generate_dataset(generated_sequences, clean_file, label, cwd, CLEAN_app_path)

    df = pd.DataFrame(data)
 
    len_all = len(df)
    # Pick the sequences w/ the highest 10% reward  (200 seqs)
    df = df.sort_values(by=['weight'], ascending=False)[['sequence', 'seq_name', 'weight']]
    best_sequences = df.iloc[:201,:]['sequence'].to_list() ## brute force it to be 200 seqs (reproducibility)

    # Fine-tune the model with the selected sequences 
    validation_split_percentage = 10
    random.shuffle(best_sequences)
    print(f'{len(best_sequences)} sequences out of {len_all} ready to finetune.')

    # Free some memory
    del df

    # load pretrained ZymCTRL-zfns tokenizer to produce training and evaluation datasets 
    tokenizer = AutoTokenizer.from_pretrained('AI4PD/ZymCTRL')

    #the objective is to get here strings, that when tokenized, would span a length of 1024, for each sequence group its length and untokenized string
    processed_dataset = []
    for i in best_sequences:
        # length of the control code
        sequence = i.strip()
        separator = '<sep>'
        control_code_length = len(tokenizer(label+separator)['input_ids'])
        available_space = 1021 - control_code_length # It is not 1024 because '<|endoftext|>', and start and end
        # Option 1: the sequence is larger than the available space (3-4% of sequences)
        if len(sequence) > available_space:
            total_length = control_code_length + len(sequence[:available_space]) + 1
            seq = f"{label}{separator}{sequence[:available_space]}<|endoftext|>"
            processed_dataset.append((total_length, seq))
        # Option 2 & 3: The sequence fits in the block_size space with or without padding
        else:
            total_length = control_code_length + len(sequence) + 3
            # in this case the sequence does not fit with the start/end tokens
            seq = f"{label}{separator}<start>{sequence}<end><|endoftext|>"
            processed_dataset.append((total_length, seq))

    grouped_dataset=dict(enumerate(grouper(processed_dataset),1))

    # Write file out for the tokenizer to read
    with open(f"{cwd}{label}_processed.txt", 'w') as fn: 
        for key,value in grouped_dataset.items():
            padding_len = 1024-len(tokenizer(value)['input_ids'])
            padding = "<pad>"*padding_len
            fn.write(value+padding)
            fn.write("\n")

    ##TOKENIZE
    # adapted from the trainer file
    data_files = {}
    dataset_args = {}

    data_files["train"] = f'{cwd}{label}_processed.txt'
    extension = "text"
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir='.', **dataset_args)

    raw_datasets["train"] = load_dataset(extension,
                    data_files=data_files,
                    split=f"train[{validation_split_percentage}%:]",
                    cache_dir='.',
                    **dataset_args,)

    raw_datasets["validation"] = load_dataset(extension,
                                            data_files=data_files,
                                            split=f"train[:{validation_split_percentage}%]",
                                            cache_dir='.',
                                            **dataset_args,)

    tokenized_datasets = raw_datasets.map(tokenize_function,
                                        batched=True,
                                        num_proc=32,
                                        remove_columns=['text'],
                                        load_from_cache_file = False,
                                        desc="Running tokenizer on dataset",
                                    )
    block_size = 1024
    lm_datasets = tokenized_datasets.map(group_texts,
                                    batched=True,
                                    num_proc=124,
                                    load_from_cache_file=False,
                                    desc=f"Grouping texts in chunks of {block_size}",
                                )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    train_dataset.save_to_disk(f'{cwd}dataset/train2')
    eval_dataset.save_to_disk(f'{cwd}dataset/eval2')


    # Free CUDA memory
    del tokenizer, train_dataset, eval_dataset, tokenized_datasets, lm_datasets, raw_datasets, best_sequences, grouped_dataset, processed_dataset


    # Finetune the model
    print('starting finetune')
    # subprocess.run(["python", "/users/nferruz/martigues/self_training/finetuner.py", "--tokenizer_name", model_savedir+f'model_{n_iteration-1}/'+checkpoint, "--do_train", "--do_eval", "--load_best_model_at_end", "--output_dir", f"{model_savedir}/model_{n_iteration}", "--evaluation_strategy", "steps", "--eval_steps", "10", "--logging_steps", "5", "--save_steps", "500", "--num_train_epochs", "28", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "1", "--cache_dir", ".", "--save_total_limit", "1", "--learning_rate", "0.8e-04", "--dataloader_drop_last", "True", "--model_name_or_path", model_savedir+f'model_{n_iteration-1}/'+checkpoint])
    if iteration_num == 1: 
        subprocess.run(["python", f"{cwd}scripts/finetuner.py", "--tokenizer_name", "AI4PD/ZymCTRL", "--model_name_or_path", "AI4PD/ZymCTRL", "--load_best_model_at_end", "--do_train", "--do_eval", "--output_dir", f"{cwd}/models/{label}_model{iteration_num}", "--evaluation_strategy", "steps", "--eval_steps", "10", "--logging_steps", "2", "--save_steps", "10", "--num_train_epochs", "25", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "1", "--cache_dir", ".", "--learning_rate", "0.8e-06", "--dataloader_drop_last", "True", "--save_total_limit", "1"])
    else: 
        checkpoint_folder = [x for x in os.listdir(f'{cwd}models/{label}_model{iteration_num-1}') if 'checkpoint' in x][0] ## changed 8/10/2024
        subprocess.run(["python", f"{cwd}scripts/finetuner.py", "--tokenizer_name", f"{cwd}models/{label}_model{iteration_num-1}/{checkpoint_folder}", "--model_name_or_path", f"{cwd}models/{label}_model{iteration_num-1}/{checkpoint_folder}", "--load_best_model_at_end", "--do_train", "--do_eval", "--output_dir", f"{cwd}models/{label}_model{iteration_num}", "--evaluation_strategy", "steps", "--eval_steps", "10", "--logging_steps", "2", "--save_steps", "10", "--num_train_epochs", "25", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "1", "--cache_dir", ".", "--learning_rate", "0.8e-06", "--dataloader_drop_last", "True", "--save_total_limit", "1"])
    print(f'round {iteration_num} of finetuning performed')
