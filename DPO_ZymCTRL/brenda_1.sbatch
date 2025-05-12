#!/bin/bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --mem=300GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate
python brenda.py --min_length 180 --max_length 260 --output_path brenda/180_260_sequences.csv