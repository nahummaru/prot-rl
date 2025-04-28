# ZymCTRL DPO Training Pipeline

This project implements a training pipeline for fine-tuning the ZymCTRL model using Direct Preference Optimization (DPO) for generating enzyme sequences with controlled stability properties.

## Overview

The pipeline consists of three main components:

1. **Sequence Generation** (`generate_training_data.py`):
   - Generates enzyme sequences using the ZymCTRL model
   - Computes stability scores for generated sequences
   - Organizes sequences into high, medium, and low stability categories
   - Supports tagged runs for experiment tracking

2. **Stability Assessment** (`stability.py`):
   - Implements ESM-IF1 based stability scoring
   - Provides both raw inverse folding scores and deltaG estimates
   - Handles batch processing of sequences

3. **Model Training** (`lightning_trainer.py`):
   - Implements both standard fine-tuning (SFT) and DPO training
   - Uses PyTorch Lightning for robust training
   - Supports wandb logging and checkpointing
   - Handles both single sequence and paired sequence training

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies (recommended to use a virtual environment)
pip install torch transformers pytorch-lightning wandb pandas tqdm
```

## Usage

### 1. Generate Training Data

```bash
python generate_training_data.py \
    --iteration_num 0 \
    --ec_label "4.2.1.1" \
    --n_batches 10 \
    --tag "experiment1"
```

Parameters:
- `iteration_num`: Training iteration (0 for initial run)
- `ec_label`: EC number for the target enzyme
- `n_batches`: Number of sequence batches to generate
- `tag`: Optional identifier for the run

### 2. Train the Model

```bash
python lightning_trainer.py \
    --model_name "AI4PD/ZymCTRL" \
    --train_data "training_data_iteration0/sequences.csv" \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --training_mode "dpo" \
    --output_dir "checkpoints"
```

Parameters:
- `model_name`: Base model to fine-tune
- `train_data`: Path to training data
- `training_mode`: "sft" or "dpo"
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `output_dir`: Directory for saving checkpoints

## Data Organization

The pipeline organizes generated sequences and training data as follows:

```
training_data_iteration{N}[_tag]/
├── sequences.csv          # Full dataset with stability scores
├── sequences.json        # Raw generated sequences
├── stability_high.fasta  # High stability sequences
├── stability_low.fasta   # Low stability sequences
└── stability_medium.fasta # Medium stability sequences
```

## Model Training Modes

1. **Standard Fine-tuning (SFT)**:
   - Traditional language model fine-tuning
   - Uses stability tags for conditional generation

2. **Direct Preference Optimization (DPO)**:
   - Trains on paired sequences
   - Optimizes for stability preferences
   - Uses beta parameter to control preference strength

## Checkpointing and Recovery

The pipeline includes automatic checkpointing and recovery features:
- Saves model checkpoints during training
- Supports resuming from interruptions
- Maintains separate checkpoints for different experimental runs

## Monitoring

Training progress can be monitored using:
- Weights & Biases integration for experiment tracking
- Local logging of loss metrics and stability distributions
- Checkpoint validation metrics

## License

[Add License Information]

## Citation

If you use this code, please cite:
[Add Citation Information] 