<img src="https://github.com/user-attachments/assets/6857cfe5-8b43-4a7c-aeea-e1e55eb04c73" width="600"/>

# ProtRL: Direct Preference Optimization for Protein Language Models

This is the repository for the paper [*Guiding Generative Protein Language Models with Reinforcement Learning*](https://arxiv.org/abs/2412.12979). ProtRL is a Reinforcement Learning (RL) framework for autoregressive protein language models (pLMs). In this repository, you will find the scripts used for all experiments in the paper (`Experiments`) and a minimal implementation of ProtRL ready to work with models like ZymCTRL, ProtGPT2, and REXzyme.

This implementation is modular and easily extendable to other pLMs and custom reward functions. We also provide utilities for training with Direct Preference Optimization (DPO), including support for different loss modes: `paired`, `ranked`, and `weighted`.

---

## Table of Contents
- [About ProtRL](#about-protrl)
- [Installation](#installation)
- [Example](#example)
- [General Usage](#general-usage)
- [Troubleshooting](#troubleshooting)
- [Work in Progress](#work-in-progress)
- [References](#references)
- [Citation](#citation)

---

## About ProtRL

`DPO_pLM.py` is the main training script. It supports the following flags:

- `--beta`: Set the inverse temperature for DPO.
- `--model_directory`: Specify the path to your model (local or Hugging Face, e.g., `AI4PD/ZymCTRL`).
- `--mode`: Choose one of three DPO loss types:
  - `paired`: Train on pairwise preference data.
  - `ranked`: Train on ordered lists of sequences by reward.
  - `weighted`: Train on sequences with scalar rewards (default and recommended).

These loss modes are adapted from [Widatalla et al., 2024](https://www.biorxiv.org/content/10.1101/2024.05.20.595026v1). See the Methods section of the [paper](https://arxiv.org/abs/2412.12979) for full details.

> ⚠️ Note: DPO weights assume higher = better. If your reward function is to be minimized (e.g., energy), remember to multiply by -1.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AI4PDLab/ProtRL.git
cd ProtRL
pip install -r requirements.txt
