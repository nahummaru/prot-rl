#!/bin/bash

# Enable PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Testing training data generation"

# Generate a small test set (2 batches of 5 sequences each)
python generate_training_data.py \
    --iteration_num 0 \
    --ec_label "4.2.1.1" \
    --n_batches 1

echo "Test completed" 