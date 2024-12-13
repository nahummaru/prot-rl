#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_template_ranked
#SBATCH --time=24:00:00

# change this configuration to run on your GPU (80GB) configuration
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80

##################################
# make bash behave more robustly #
##################################
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

label="4.2.1.1" # EC label we want to prompt ZymCTRL with 
model_directory="AI4PD/ZymCTRL" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="paired" # choose between paired, ranked and weighted 

echo DPO_pLM for the enzyme class $label, with $DPO_mode mode

# establish the number of iterations you want to do with DPO_pLM
for i in $(seq 0 10);

do

    echo Starting iteration $i
    # Train the model, 30 epochs each, except first 
    
    if [ $i != 0 ]; then
    
      echo Train started
      python DPO_pLM.py --iteration_num $i --ec_label $label --mode $DPO_mode --model_dir $model_directory
    
    fi

    echo Sequence generation started
    # Generate the sequences
    python seq_gen.py --iteration_num $i --ec_label $label
        
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
