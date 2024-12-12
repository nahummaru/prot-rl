#!/bin/bash

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# time limit in seconds
#SBATCH --time=12:00:00

# queue
#SBATCH --qos=normal

# resources
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=40000

# job name
#SBATCH --job-name sft_TMscore_4.2.1.1

#################
# start message #
#################
start_epoch=`date +%s`
echo [$(date +"%Y-%m-%d %H:%M:%S")] starting on $(hostname)

##################################
# make bash behave more robustly #
##################################
set -e
set -u
set -o pipefail

###################
# set environment #
###################
# activate your environment 
# load python module 
your_local_foldseek_path=""
export PATH=$your_local_foldseek_path/foldseek/bin/:$PATH

###############
# run command #
###############

#Run the script for 25 rounds of RL (1000 feedback points)

folder_path="" # in which you want the results to be in 
label="4.2.1.1" # EC label of the CA we want to change folds from beta to alpha 
echo self-training of ZymCTRL for TMscore with "${label}" started

for i in $(seq 0  30);
do
    echo Starting iteration $i
    # Train the model, 30 epochs each, except first 
    if [ $i != 0 ]; then
    echo Train started
    python "${folder_path}scripts/self_train_TMscore.py" --iteration_num "${i}" --label "${label}" --cwd "${folder_path}"
    else
    # create the file tree architechture to save the different results 
    mkdir generated_sequences
    mkdir PDB
    mkdir TMscores
    mkdir models
    mkdir dataset
    fi

    echo Sequence generation started
    # Generate the sequences, 2000 each round
    python "${folder_path}scripts/seq_gen.py" --iteration_num "${i}" --label "${label}" --cwd "${folder_path}"
    
    # Fold the sequences with ESM fold
    echo Folding started
    python "${folder_path}scripts/esmfold.py" --iteration_num "${i}" --label "${label}" --cwd "${folder_path}"
    
    # Calculate TM Score
    echo foldseek started for 2vvb # superimposing to alpha CA fold 
    foldseek easy-search "${folder_path}"PDB/"${label}"_output_iteration"${i}" "${folder_path}"2vvb.pdb "${folder_path}"TMscores/"${label}"_TM_iteration"${i}" "${folder_path}"tmp --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0
    echo foldseek started for 1i6p # superimposing to beta CA fold 
    foldseek easy-search "${folder_path}"PDB/"${label}"_output_iteration"${i}" "${folder_path}"1i6p.pdb "${folder_path}"TMscores/"${label}"_beta_TM_iteration"${i}" "${folder_path}"tmp --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0

done

###############
# end message #
###############
cgroup_dir=$(awk -F: '{print $NF}' /proc/self/cgroup)
peak_mem=`cat /sys/fs/cgroup$cgroup_dir/memory.peak`
echo [$(date +"%Y-%m-%d %H:%M:%S")] peak memory is $peak_mem bytes
end_epoch=`date +%s`
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname) after $((end_epoch-start_epoch)) seconds
