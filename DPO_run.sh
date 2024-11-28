#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_template_ranked
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80

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

label="4.2.1.1"
model_directory="AI4PD/ZymCTRL"
DPO_mode="paired"

echo RL for the enzyme class $label, with mode $DPO_mode


for i in $(seq 11 30);

do

    echo Starting iteration $i
    # Train the model, 30 epochs each, except first 
    
    if [ $i != 0 ]; then
    
      echo Train started
      python DPO.py --iteration_num $i --label $label --mode $DPO_mode --model_dir $model_directory
    
    fi

    echo Sequence generation started
    # Generate the sequences
    python seq_gen.py --iteration_num $i --label $label
        
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
