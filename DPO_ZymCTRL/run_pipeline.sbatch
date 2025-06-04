#!/bin/bash

#SBATCH --partition=hsu_gpu_priority
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate
# python train_iterative.py --n_iterations 1 --learning_rate 1e-5 --num_epochs 5 --warmup_steps 100 --weight_decay 0.01 --tag "brenda_sft_only_high_stability_ablation" --training_mode sft --train_data_csv "brenda/train_sequences.csv" --val_data_csv "brenda/val_sequences.csv" --batch_size 32
python -u train_iterative.py --n_iterations 1 --learning_rate 1e-5 --num_epochs 5 --warmup_steps 100 --weight_decay 0.01 --tag "talal-dpo-2" --training_mode dpo --train_data_csv "brenda/train_sequences.csv" --val_data_csv "brenda/val_sequences.csv" --batch_size 32 --stability_threshold 2 --n_pairs_to_sample 20000 --max_sampling_attempts 100000
# python -u train_iterative.py --n_iterations 1 --learning_rate 1e-5 --num_epochs 5 --warmup_steps 100 --weight_decay 0.01 --tag "epoch4-sft-dpo_every_epoch" --training_mode dpo --train_data_csv "brenda/train_sequences.csv" --val_data_csv "brenda/val_sequences.csv" --batch_size 32 --initial_model "checkpoints_iteration0_brenda_sft/epoch=4-val_loss=0.44.ckpt" --use_control_tags --n_pairs_to_sample 20000 --max_sampling_attempts 50000 --stability_threshold 3

