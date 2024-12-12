#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_CA_TM_aln
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

#export http_proxy=http://proxy:80
#export https_proxy=http://proxy:80
    
###############
# run command #
###############

label="4.2.1.1"

echo RL for the enzyme class $label

for i in $(seq 8 30);
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
        
    # Fold the sequences with ESM fold
    echo Folding started
    python ESM_Fold.py --iteration_num $i --label $label
      
      # Calculate TM Score
    echo foldseek started for 2vvb
    export PATH=./foldseek/bin/:$PATH
    foldseek easy-search output_iteration$i/PDB  '2vvb.pdb' alpha_${label}_TM_iteration$((i)) tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0
      
    echo foldseek started for 1i6p
    foldseek easy-search output_iteration$i/PDB  '1i6p.pdb' beta_${label}_TM_iteration$((i)) tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0
      
    
    # Calculate aligment and clusters
    echo Aligments and cluster 
    export PATH=./mmseqs/bin/:$PATH
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.9_seq_gen_${label}_iteration$((i)) tmp --min-seq-id 0.9
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.5_seq_gen_${label}_iteration$((i)) tmp --min-seq-id 0.5
    mmseqs easy-search  seq_gen_${label}_iteration$((i)).fasta ./brenda_dataset/database_${label}.fasta alignment/alnResult_seq_gen_${label}_iteration$((i)).m8 tmp    
      


done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)

