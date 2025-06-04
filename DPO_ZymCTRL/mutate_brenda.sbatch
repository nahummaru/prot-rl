#!/bin/bash
#SBATCH --partition=hsu_gpu_priority
#SBATCH --mem=300GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate

python mutate_brenda_sequences.py --input_csv test_brenda_sequences.csv --output_path test_output.csv --n_mutations 1 --scoring_method rosetta --batch_size 8 