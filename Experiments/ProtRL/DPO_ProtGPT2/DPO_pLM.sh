#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=ProtWrap
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

module load python/3.12-conda
module load cuda/12.6.1

conda activate BindCraft

###############
# run command #
###############

label="egfr" # use lower caps pls 
model_directory="/home/woody/b114cb/b114cb23/models/ProtGPT2" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="weighted" # choose between paired, ranked and weighted 
hotspot_residues=""

echo DPO_pLM for the enzyme class $label, with $DPO_mode mode

# establish the number of iterations you want to do with DPO_pLM
for i in $(seq 0 30);

do

    echo Starting iteration $i

    if [ $i != 0 ]; then
    
      echo Train started
      python "DPO_pLM.py" --iteration_num $i --label $label --mode $DPO_mode --model_dir $model_directory
    
    fi 

    echo Sequence generation started
    # Generate the sequences
    python "seq_gen.py" --iteration_num $i --label $label  --model_dir $model_directory
    
    echo "Folding started"
    apptainer exec \
        --nv \
        --bind /home/woody/b114cb/b114cb23/ProtWrap/:/root/af_input \
        --bind /home/woody/b114cb/b114cb23/ProtWrap/:/root/af_output \
        --bind /home/woody/b114cb/b114cb23/Filippo/AF3_model/:/root/models \
        --bind /anvme/data/alphafold3/databases/:/root/public_databases \
        /anvme/data/alphafold3/alphafold3-20250125.sif\
        python /home/woody/b114cb/b114cb23/alphafold3/run_alphafold.py\
        --input_dir=./alphafold3_input_iteration_"$i" \
        --output_dir=./alphafold_output_iteration"$i" \
        --run_data_pipeline=False \
        --model_dir=/root/models
    
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
