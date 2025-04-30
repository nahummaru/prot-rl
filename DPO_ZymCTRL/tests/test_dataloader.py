import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from lightning_trainer import ZymCTRLDataset
import pandas as pd

def print_sample_info(sample, mode):
    """Helper function to print sample information"""
    print(f"\n=== {mode} Sample ===")
    
    if mode == "SFT":
        # Print raw data
        print(f"EC Label: {sample['ec_label']}")
        print(f"Sequence: {sample['sequence'][:50]}...")  # First 50 chars
        print(f"Stability Score: {sample['stability_score']:.2f}")
        print(f"Stability Level: {sample['stability_level']}")
        
        # Print tokenized data shapes
        print("\nTokenized Data Shapes:")
        print(f"input_ids: {sample['input_ids'].shape}")
        print(f"attention_mask: {sample['attention_mask'].shape}")
        print(f"labels: {sample['labels'].shape}")
        
    else:  # DPO mode
        # Print raw data
        print(f"EC Label: {sample['ec_label']}")
        print(f"Prefer Stable: {sample['prefer_stable']}")
        print(f"Chosen Score: {sample['chosen_score']:.2f}")
        print(f"Rejected Score: {sample['rejected_score']:.2f}")
        
        # Print tokenized data shapes
        print("\nTokenized Data Shapes:")
        print(f"chosen_input_ids: {sample['chosen_input_ids'].shape}")
        print(f"chosen_attention_mask: {sample['chosen_attention_mask'].shape}")
        print(f"rejected_input_ids: {sample['rejected_input_ids'].shape}")
        print(f"rejected_attention_mask: {sample['rejected_attention_mask'].shape}")

def test_dataloader(data_path, model_name, mode="sft", batch_size=4):
    print(f"\n=== Testing {mode.upper()} Dataset ===")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset
    dataset = ZymCTRLDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        training_mode=mode.lower(),
        max_length=512
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for easier debugging
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a few samples
    print("\n--- Individual Samples ---")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        import pdb; pdb.set_trace()
        print_sample_info(sample, mode.upper())
        
    # Test batch loading
    print("\n--- Testing Batch Loading ---")
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    
    if mode.lower() == "sft":
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
    else:
        print(f"Batch chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
        print(f"Batch rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
        print(f"Number of prefer_stable True: {batch['prefer_stable'].sum().item()}")
        
    print("\nDataloader test completed successfully!")

def main():
    # Test parameters
    data_path = "training_data_iteration0/sequences.csv"  # Update path as needed
    model_name = "AI4PD/ZymCTRL"  # Using the base model
    batch_size = 4
    
    # Test both modes
    test_dataloader(data_path, model_name, mode="sft", batch_size=batch_size)
    test_dataloader(data_path, model_name, mode="dpo", batch_size=batch_size)

if __name__ == "__main__":
    main() 