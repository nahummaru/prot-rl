#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_Binder_FT_EGF
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40


set -e
set -u
set -o pipefail


###################
# set environment #
###################
module load python

source .env/bin/activate

###############
# run command #
###############

#python dataset_creation.py

python 5.run_clm-post.py --tokenizer_name ./ZymCTRL --do_train --do_eval --output_dir output_iteration0 --evaluation_strategy steps --eval_steps 10 --logging_steps 5 --save_steps 500 --num_train_epochs 30 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --cache_dir '.' --save_total_limit 2 --learning_rate  1e-06 --dataloader_drop --model_name_or_path ./ZymCTRL
python seq_gen.py --label '1.3.3.18' --iteration_num 1
###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
