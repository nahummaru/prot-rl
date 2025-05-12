#!/bin/bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --mem=300GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1

source ../dpo/bin/activate
python generate_synthetic.py --n_iterations 1 --learning_rate 1e-5 --num_epochs 3 --warmup_steps 100 --weight_decay 0.01 --tag "sft_labels" --training_mode sft