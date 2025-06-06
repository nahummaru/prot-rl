# Learning New Biophysical Controls in Protein Language Models via Supervised and Preference-Based Fine-Tuning

This repository accompanies the paper **"Learning New Biophysical Controls in Protein Language Models via Supervised and Preference-Based Fine-Tuning."** We explore how to steer generative protein language models using both supervised fine-tuning (SFT) and Direct Preference Optimization (DPO), focusing on thermodynamic stability as a target property.

---

## Overview

- **Model**: Based on [ZymCTRL](https://huggingface.co/AI4PD/ZymCTRL), a functionally conditioned protein language model (pLM).
- **Control Objective**: Thermodynamic stability (approximated via Rosetta total energy).
- **Methods**:
  - **SFT**: Tag-based conditioning using stability buckets (`<stability="high">`, etc.).
  - **DPO**: Preference-based fine-tuning on pairwise samples, guided by predicted stability differences.

---

## Setup

```bash
git clone https://github.com/nahummaru/prot-rl.git
cd prot-rl
pip install -r requirements.txt

## Training and Evaluation

To train our best SFT model
```
sbatch run_pipeline.sbatch
```

To run evals on that model
```
sbatch run_evals.sbatch
```

