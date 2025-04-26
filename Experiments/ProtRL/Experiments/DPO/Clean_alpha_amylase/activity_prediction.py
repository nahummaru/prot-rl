import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse

from peft import LoraConfig, inject_adapter_in_model
from datasets import Dataset


class SequenceDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.input_ids = torch.cat([seq["input_ids"] for seq in tokenized_sequences])
        self.attention_mask = torch.cat([seq["attention_mask"] for seq in tokenized_sequences])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def load_esm_model(checkpoint, num_labels, half_precision, full=False, deepspeed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels,
        torch_dtype=torch.float16 if half_precision and deepspeed else None
    )
    if full:
        return model, tokenizer

    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
    )
    model = inject_adapter_in_model(peft_config, model)
    for param_name, param in model.classifier.named_parameters():
        param.requires_grad = True
    return model, tokenizer


def save_model(model, filepath):
    non_frozen_params = {
        param_name: param
        for param_name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(non_frozen_params, filepath)


def load_model(checkpoint, filepath, num_labels=1, mixed=False, full=False, deepspeed=True):
    model, tokenizer = (
        load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)
        if "esm" in checkpoint
        else load_T5_model(checkpoint, num_labels, mixed, full, deepspeed)
    )
    non_frozen_params = torch.load(filepath)
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data
    return tokenizer, model


def generate_dataset(iteration_num, ec_label, tokenizer):
    tokenized_sequences = []
    names = []
    with open(f"seq_gen_{ec_label}_iteration{iteration_num}.fasta", "r") as f:
        rep_seq = f.readlines()
    for line in rep_seq:
        if not line.startswith(">"):
            encoded = tokenizer(
                line.strip(), max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokenized_sequences.append(encoded)
        else:
            names.append(line)
    dataset = SequenceDataset(tokenized_sequences)
    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    return test_dataloader, names


parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int, required=True)
parser.add_argument("--label", type=str, required=True)
args = parser.parse_args()

iteration_num = args.iteration_num
ec_label = args.label.strip()

checkpoint = "facebook/esm2_t33_650M_UR50D"
tokenizer, model = load_model(
    checkpoint,
    "./esm_GB1_finetuned.pth",
    num_labels=1
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

test_dataloader, names = generate_dataset(iteration_num, ec_label, tokenizer)

predictions = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions.extend(logits.squeeze().tolist())

del model
torch.cuda.empty_cache()

checkpoint = "/home/woody/b114cb/b114cb23/esm1v_t33_650M_UR90S_1"
tokenizer, model = load_model(
    checkpoint,
    "./Esm1v_GB1_finetuned.pth",
    num_labels=1
)
model.to(device)
model.eval()

predictions2 = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        predictions2.extend(logits.squeeze().tolist())

predictions = [(p1 + p2) / 2 for p1, p2 in zip(predictions, predictions2)]

out = "".join(f'{name[:-2]},{prediction}\n' for name, prediction in zip(names, predictions))
with open(f'activity_prediction_iteration{iteration_num}.txt', 'w') as f:
    f.write(out)

del model
torch.cuda.empty_cache()
