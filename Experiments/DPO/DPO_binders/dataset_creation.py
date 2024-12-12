import random
from transformers import AutoTokenizer

from datasets import load_dataset
import transformers
from transformers.testing_utils import CaptureLogger

## DEFINE THESE VARIABLES
tokenizer = AutoTokenizer.from_pretrained('./ZymCTRL')
ec_label = '1.3.3.18' # CHANGE TO YOUR LABEL
validation_split_percentage = 10 # change if you want
sequence_file = 'EGF_rep.fasta'


#Load sequences, Read source file
with open(sequence_file, 'r') as fn: #! CHANGE TO SEQUENCES.FASTA
    data = fn.readlines()
    fn.close()

# Put sequences into dictionary
sequences={}
for line in data:
    if '>' in line:
        name = line.strip()
        sequences[name] = []  #! CHANGE TO corre
        continue
    sequences[name].append(line.strip())

#Pass sequences to list and shuffle their order randomly
sequences_list = [(key,value[0]) for key,value in sequences.items()]
random.shuffle(sequences_list)

#the objective is to get here strings, that when tokenized, would span a length of 1024.
#for each sequence group its length and untokenized string
print("procesing dataset")
processed_dataset = []
for i in sequences_list:
    # length of the control code
    sequence = i[1].strip()
    separator = '<sep>'
    control_code_length = len(tokenizer(ec_label+separator)['input_ids'])
    available_space = 1021 - control_code_length # It is not 1024 because '<|endoftext|>', and start and end

    # Option 1: the sequence is larger than the available space (3-4% of sequences)
    if len(sequence) > available_space:
        total_length = control_code_length + len(sequence[:available_space]) + 1
        seq = f"{ec_label}{separator}{sequence[:available_space]}<|endoftext|>"
        processed_dataset.append((total_length, seq))

    # Option 2 & 3: The sequence fits in the block_size space with or without padding
    else:
        total_length = control_code_length + len(sequence) + 3
        # in this case the sequence does not fit with the start/end tokens
        seq = f"{ec_label}{separator}<start>{sequence}<end><|endoftext|>"
        processed_dataset.append((total_length, seq))

# Group sequences
def grouper(iterable):
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

print("grouping processed dataset")
grouped_dataset=dict(enumerate(grouper(processed_dataset),1))

# Write file out for the tokenizer to read
fn = open(f"{ec_label}_processed.txt",'w')
for key,value in grouped_dataset.items():
    padding_len = 1024 - len(tokenizer(value)['input_ids'])
    padding = "<pad>"*padding_len
    fn.write(value+padding)
    fn.write
    fn.write("\n")
fn.close()

##TOKENIZE
# adapted from the trainer file
data_files = {}
dataset_args = {}

data_files["train"] = f"{ec_label}_processed.txt"
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

def tokenize_function(examples):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples["text"])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
        )
    return output

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=32,
    remove_columns=['text'],
    load_from_cache_file = False,
    desc="Running tokenizer on dataset",
)

block_size = 1024
def group_texts(examples):
    # Concatenate all texts.
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

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=124,
    load_from_cache_file=False,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

train_dataset.save_to_disk('./dataset/train2')
eval_dataset.save_to_disk('./dataset/eval2')
