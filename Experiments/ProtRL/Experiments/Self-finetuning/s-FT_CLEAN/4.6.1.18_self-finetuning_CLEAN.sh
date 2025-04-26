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
#SBATCH --job-name sft_CLEAN_4.6.1.18

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

###############
# run command #
###############
# in the folder path you should have a folder named "scripts" with all git scripts and files CLEAN_infer_fasta.py and 
folder_path="" # folder you want the results file tree in 
your_local_CLEAN_path="" # indicate the folder containing your copy of the CLEAN local installation 
CLEAN_app_path="$your_local_CLEAN_path/CLEAN/app/"
label="4.6.1.18"  # EC label of the enzyme with just 10% proportion of good predictions we want to increase 

echo Starting self-training for "${label}" with CLEAN reward
for i in $(seq 0  30);
do
    echo Starting iteration "${i}"
    # Train the model, 30 epochs each, except first 
    if [ $i != 0 ]; then
    echo Train started
    python "${folder_path}"scripts/self_train_CLEAN.py --iteration_num "${i}" --label "${label}" --cwd "${folder_path}" --CLEAN_app_path "${CLEAN_app_path}"
    else
    # create the file tree architechture to save the different results at iteration == 0
    mkdir generated_sequences
    mkdir PDB
    mkdir TMscores
    mkdir models
    mkdir dataset
    cp "${folder_path}scripts/CLEAN_infer_fasta.py" "${CLEAN_app_path}"
    fi 

    echo Sequence generation started
    # Generate the sequences, 2000 each round
    python "${folder_path}"scripts/seq_gen.py --iteration_num "${i}" --label "${label}" --cwd "${folder_path}"
    
    # Determine the EC number
    echo estimation of the E.C number for each fasta
    cp "${folder_path}"generated_sequences/seq_gen_"${label}"_iteration"${i}".fasta "${CLEAN_app_folder_path}"data/inputs
    cd "${CLEAN_app_folder_path}"
    python "${CLEAN_app_folder_path}"CLEAN_infer_fasta.py --iteration_num "${i}" --label $label
    
    # Get esm embeddings 
    echo Retriving esm embeddings
    python "${CLEAN_app_folder_path}"esm/scripts/extract.py esm1b_t33_650M_UR50S "${CLEAN_app_folder_path}"data/inputs/seq_gen_"${label}"_iteration"${i}".fasta "${CLEAN_app_folder_path}"data/esm_data --include mean
    
    cd -
    cp "${CLEAN_app_folder_path}"results/inputs/seq_gen_"${label}"_iteration"${i}"_maxsep.csv "${folder_path}"clean/seq_gen_"${label}"_iteration"${i}"_maxsep.csv
    
done

###############
# end message #
###############
cgroup_dir=$(awk -F: '{print $NF}' /proc/self/cgroup)
peak_mem=`cat /sys/fs/cgroup$cgroup_dir/memory.peak`
echo [$(date +"%Y-%m-%d %H:%M:%S")] peak memory is $peak_mem bytes
end_epoch=`date +%s`
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname) after $((end_epoch-start_epoch)) seconds
