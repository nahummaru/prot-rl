#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_Binder_t10_zymctrl
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

set -e
set -u
set -o pipefail


###################
# set environment #
###################
module load python
module load cuda/11.8.0 
module load cudnn/8.9.6.50-11.x
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
    
###############
# run command #
###############

label="1.3.3.18"

echo RL for the enzyme class $label

for i in $(seq  30);
do
    
    echo Starting iteration $i
    # Train the model, 30 epochs each, except first 
    if [ $i != 0 ]; then
    
      echo Train started
      python DPO_weight2.py --iteration_num $i --label $label

    fi
      
      echo Sequence generation started
      # Generate the sequences
      python seq_gen.py --iteration_num $i --label $label
      
      # Fold the sequences with Alphafold
      module unload python 
      module load localcolabfold/1.5.2
      echo Folding started
      colabfold_batch \
              --num-recycle 1 \
              --templates \
              --msa-mode single_sequence \
              --num-models 1 \
              --overwrite-existing-results \
              seq_gen_${label}_iteration$((i)).fasta\
              AF/iteration$((i)) 


      module unload localcolabfold/1.5.2
      module load python 
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
