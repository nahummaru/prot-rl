#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_esmv1
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
directory=${PWD##*/}
directory_all=${PWD}

echo RL for the enzyme class $label

echo 
for i in $(seq 0 30);

do

    echo Starting iteration $i
    # Train the model, 30 epochs each, except first 
    
    if [ $i != 0 ]; then
    
      echo Train started
      python DPO_weight2.py --iteration_num $i --label $label   
      echo Sequence generation started
  
    fi

    # Generate the sequences
    python seq_gen.py --iteration_num $i --label $label    
        
    # Fold the sequences with ESM fold
    echo Folding started
    python ESM_Fold.py --iteration_num $i  --label $label

    cd /home/woody/b114cb/b114cb23/DPO/DPO_Clean/CLEAN/app

    # Get esm embeddings 
    echo Retriving esm embeddings
    python esm/scripts/extract.py esm1b_t33_650M_UR50S data/inputs/seq_gen_${label}_iteration$((i)).fasta data/esm_data --include mean
      
    # Determine the EC number
    echo estimation of the E.C number for each fasta
    python CLEAN_infer_fasta.py --iteration_num $((i)) --label $label
      
    cd -
      
    cp "./CLEAN/app/results/inputs/seq_gen_${label}_iteration$((i))_maxsep.csv" seq_gen_${label}_iteration$((i))_maxsep.csv
    
    # Determine the esm1v value
    cd ,/protein_gibbs_sampler
    module load cuda/12.4.1
    source  .gibbs_sampler/bin/activate
    cd ./protein_gibbs_sampler/src/pgen
    echo the directory is ./${directory}/seq_gen_${label}_iteration$((i)).fasta
    python likelihood_esm.py -i ./${directory}/seq_gen_${label}_iteration$((i)).fasta -o ./${directory}/metrics_${label}_iteration$((i)).txt --model esm1v --csv --device gpu --score_name esm1v
    module unload cuda
    source deactivate
    source .env/bin/activate
    cd $directory_all

    # Calculate TM Score
    echo foldseek started for 11ba
    export PATH=./foldseek/bin/:$PATH
    foldseek easy-search output_iteration$((i))/PDB  '1i6p.pdb' ${label}_TM_iteration$((i)) tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0
    
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
