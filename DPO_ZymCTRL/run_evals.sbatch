#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate

python eval_harness.py \
       --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints/brenda_epoch_0.ckpt" \
       --eval_type controllability \
       --output_dir "brenda_results" \
       --num_samples 100 \
       --batch_size 10