#!/bin/bash

#SBATCH --partition=hsu_gpu_priority
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate

python eval_harness.py \
       --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_brenda_sft_15/epoch=2-val_loss=0.49.ckpt" \
       --eval_type controllability \
       --output_dir "brenda_results_sft_15" \