![Git_front](https://github.com/user-attachments/assets/43fb4f1c-471f-4178-a1e5-e323ef36f533)

# DPO_pLM: Direct Preference Optimization for Protein Language Models

This is the official repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](). DPO_pLM is a Reinforcement Learning (RL) framework for protein sequence optimization for autoregressive protein Language Models (pLMs). In this repository you will find the scripts used for the experiments found on the paper (`Experiments`) and a basic implementation of DPO_pLM ready to work with ZymCTRL and optimize sequences to a desired length. This implementation is minimal and easily amenable other pLMs and custom reward functions. 

## About DPO_pLM

`DPO_pLM` is the main Python script for our project. It supports the following flags:

- `--beta`: Specify the beta value.
- `--model_directory`: Specify the local model or the HF name (e.g AI4PD/ZymCTRL or /your/path/ZymCTRL)
- `--mode`: Choose the mode for experiments. Available options:
  - `paired`: the loss function will take in an ordered pair of sequences responses with different rewards and train the model to give preference to the sequence with better reward score. 
  - `ranked`: the loss function will take in an ordered set of sequences responses by their reward score and train the model to give preference to the sequences in the ranking order.
  - `weighted`: the loss function will take in a set of sequence with their corresponding reward score and will train the model’s distribution of likelihood over the sequences to match the relative distribution of reward of the sequences derived from the softmax of their scalar labels.

This 3 different loss functions were adapted from the firsts described in [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1.abstract). You can find detailed explanations for each loss function and its changes in formulation in the Methods section of the [paper](). 

## Installation

The software needed to run DPO_pLM can be found in `requeriments.txt`. To set up the environment, execute the following comand inside your desired working environment:

```bash
git clone https://github.com/AI4PDLab/DPO_pLM.git
cd DPO_pLM
pip install -r requirements.txt
```
This work has been developed and tested on Python 3.10.4.

## Example 

DPO_pLM is reported as a very simple script with the objective of increasing the lenght over the different iterations. In the `Experiments` folder you can find the scripts for experiments that implement more complex scoring functions such as protein folds, functional annotation of enzymes and experimental data. If you are interested in optimizing for other protein features, you can use `DPO_pLM.py` as a template for your custom experiments.

First of all, you will need to set up ZymCTRL or the pLM of your choice. In our case we downloaded the [HuggingFace's ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL) repository locally or use it directly from the repo, taking advantage from Huggingface's `transformers` API (AI4PD/ZymCTRL). 

To give an overview of how the different loss functions (or modes) differ, the 3 modes have been ran to generate sequences of around 600 amino acids. With this really simple task we can see that the 3 modes achieve the desired goal in just few iterations.

![image](https://github.com/user-attachments/assets/b408b256-0697-45b2-a396-2312f87f1ed8)

Note that in this case, the objective is to maximise the weight (sequence lenght), thus the weight must be multiplied by (-1)

To reproduce the experiments of our paper, you can find all the scripts in the `Experiments` folder. Given the size and computational needs of pLMs, each one of the experiments were executed in one H100 GPU, with differing times of execution. All the parameters and external data used in the experiments can be found in this repo. The `.sh` scripts can be executed from the same folder to conduct each experiment, they have been built to work on a SLURM based cluster, given the need of GPU-intensive computing. To reproduce the results run: 

```bash
bash experiment_name.sh
```
or 
```bash 
sbatch experiment_name.sh
```
Replace `experiment_name` with the desired experiment script path. Each experiment will produce, fold and calculate statistics for each considered feature.

## Troubleshooting

Refer to the documentation for more details on how to configure and run your experiments.

Feel free to contribute or raise issues if you encounter any problems! We are working to make it more accessible and detailed

## References
- ESM1v: "Language models enable zero-shot prediction of the effects of mutations on protein function" Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives; doi: https://doi.org/10.1101/2021.07.09.450648. Computed using https://github.com/seanrjohnson/protein_gibbs_sampler/
- ProteinMPNN: "Robust deep learning–based protein sequence design using ProteinMPNN", J. Dauparas et al. Science378,49-56(2022).DOI:10.1126/science.add2187
- CLEAN: "Enzyme function prediction using contrastive learning". Science379,1358-1363(2023). DOI:10.1126/science.adf2465, GitHub: "https://github.com/tttianhao/CLEAN?tab=readme-ov-file"

