#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=ZymCtrl_RL_Extension
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

# Parameters
EC_LABEL="4.2.1.1"  # Example EC number
MODEL_DIR="AI4PD/ZymCTRL"  # Base ZymCtrl model
DPO_MODE="weighted"  # Choose between paired, ranked, weighted
NUM_ITERATIONS=10
EVAL_INTERVAL=2  # Evaluate every N iterations

echo "Starting ZymCtrl RL Extension Experiment"
echo "EC Label: $EC_LABEL"
echo "DPO Mode: $DPO_MODE"

# Create directories for results
mkdir -p results/stability
mkdir -p results/ec_eval
mkdir -p results/multi_tag

for i in $(seq 0 $NUM_ITERATIONS); do
    echo "Starting iteration $i"
    
    if [ $i != 0 ]; then
        echo "Training started"
        # Train with stability control tag
        python DPO_pLM.py \
            --iteration_num $i \
            --ec_label $EC_LABEL \
            --mode $DPO_MODE \
            --model_dir $MODEL_DIR \
            --control_tag stability
        
        # Train with multi-tag (EC + stability)
        python DPO_pLM.py \
            --iteration_num $i \
            --ec_label $EC_LABEL \
            --mode $DPO_MODE \
            --model_dir $MODEL_DIR \
            --control_tag multi
    fi
    
    echo "Sequence generation started"
    # Generate sequences with different control tags
    python seq_gen.py \
        --iteration_num $i \
        --ec_label $EC_LABEL \
        --control_tag none  # EC only
    
    python seq_gen.py \
        --iteration_num $i \
        --ec_label $EC_LABEL \
        --control_tag stability  # EC + stability
    
    python seq_gen.py \
        --iteration_num $i \
        --ec_label $EC_LABEL \
        --control_tag multi  # EC + stability + other tags
    
    echo "Stability evaluation started"
    # Evaluate stability of generated sequences
    python stability.py \
        --iteration_num $i \
        --ec_label $EC_LABEL \
        --output_dir results/stability
    
    # Evaluate EC-specific capabilities
    if [ $((i % EVAL_INTERVAL)) -eq 0 ]; then
        echo "EC-specific evaluation started"
        python ec_eval.py \
            --iteration_num $i \
            --ec_label $EC_LABEL \
            --output_dir results/ec_eval
    fi
    
    # Evaluate multi-tag performance
    if [ $((i % EVAL_INTERVAL)) -eq 0 ]; then
        echo "Multi-tag evaluation started"
        python multi_tag_eval.py \
            --iteration_num $i \
            --ec_label $EC_LABEL \
            --output_dir results/multi_tag
    fi
    
    # Save model checkpoints
    mkdir -p checkpoints/iteration_$i
    cp -r output_iteration$i/* checkpoints/iteration_$i/
done

###############
# end message #
###############
echo "[$(date +"%Y-%m-%d %H:%M:%S")] Experiment completed on $(hostname)" 