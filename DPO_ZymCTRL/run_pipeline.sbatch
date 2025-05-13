#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate
python train_iterative.py --n_iterations 1 --learning_rate 1e-5 --num_epochs 5 --warmup_steps 100 --weight_decay 0.01 --tag "brenda_sft_only_high_stability_ablation" --training_mode sft --train_data_csv "brenda/train_sequences.csv" --val_data_csv "brenda/val_sequences.csv" --batch_size 32
