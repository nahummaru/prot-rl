#!/bin/bash -l

#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80

#SBATCH --job-name=DPO_Rexzyme



source /home/woody/b114cb/b114cb23/.test_env/bin/activate
module load cuda/12.6.1


#################
# Start Message #
#################

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting on $(hostname)"

##################################
# Make Bash Behave More Robustly #
##################################

set -e
set -u
set -o pipefail


model_directory="/home/woody/b114cb/b114cb23/models/checkpoint-106500"  # Path to local model or Huggingface repository
DPO_mode="weighted"
label="CA"
echo DPO_pLM with REXzyme, with $DPO_mode mode

# establish the number of iterations you want to do with DPO_pLM
for i in $(seq 1 100);

do

    echo Starting iteration $i
    
    if [ $i != 0 ]; then
    
      echo Train started
      python DPO_pLM.py --iteration_num $i --mode $DPO_mode --label $label --model_dir $model_directory
    
    fi

    echo Sequence generation started
    # Generate the sequences
    python seq_gen.py --iteration_num $i --model_dir $model_directory --label $label
        
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
