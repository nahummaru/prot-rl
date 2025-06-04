#!/bin/bash

#SBATCH --partition=hsu_gpu_priority
#SBATCH --mem=200GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

source ../dpo/bin/activate

# python eval_harness.py \
#        --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_brenda_sft_only_high_stability_ablation/epoch=4-val_loss=0.49.ckpt" \
#        --eval_type controllability \
#        --output_dir "results/brenda_sft_only_high_stability_ablation" \
#        --disable_tags

# python eval_harness.py \
#        --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_talal-dpo/epoch=0-val_loss=0.68.ckpt" \
#        --eval_type controllability \
#        --output_dir "results/talal_dpo_sft_only_high_stability_ablation_epoch0" \
#        --disable_tags

# python eval_harness.py \
#        --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_epoch4-sft-dpo/epoch=4-val_loss=0.46.ckpt" \
#        --eval_type controllability \
#        --output_dir "results/epoch4-sft-dpo"

# python eval_harness.py \
#        --model_path "AI4PD/ZymCTRL" \
#        --eval_type controllability \
#        --output_dir "results/zymctrl_baseline_no_tags" \
#        --disable_tags

# python eval_harness.py \
#        --model_path "AI4PD/ZymCTRL" \
#        --eval_type controllability \
#        --output_dir "results/zymctrl_baseline_with_tags"

# python eval_harness.py \
#        --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_epoch4-sft-dpo_every_epoch/epoch=1-val_loss=0.25.ckpt" \
#        --eval_type controllability \
#        --output_dir "results/sft-dpo-epoch1"

# python eval_harness.py \
#        --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_epoch4-sft-dpo_every_epoch/epoch=0-val_loss=0.67.ckpt" \
#        --eval_type controllability \
#        --output_dir "results/sft-dpo-epoch0"

python eval_harness.py \
       --model_path "/home/joetey/prot-rl/DPO_ZymCTRL/checkpoints_iteration0_talal-dpo-2/epoch=0-val_loss=0.33.ckpt" \
       --eval_type controllability \
       --output_dir "results/talal-dpo-3" \
       --disable_tags

