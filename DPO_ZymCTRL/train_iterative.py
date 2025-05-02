import os
import argparse
import subprocess
from pathlib import Path

"""
WORK IN PROGRESS: THIS WILL BECOME THE MAIN TRAINING SCRIPT. RUN ONLINE TRAINING

"""

def run_data_generation(iteration, ec_label, n_sequences, tag="", model_path=None):
    """Run the data generation script for current iteration"""
    # Calculate batches needed (assuming 2 sequences per batch from generate_training_data.py)
    n_batches = n_sequences
    
    cmd = [
        "python", "generate_training_data.py",
        "--iteration_num", str(iteration),
        "--ec_label", ec_label,
        "--n_batches", str(n_batches),
        "--tag", tag
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])
        
    print(f"\n=== Generating {n_sequences} sequences for iteration {iteration} ===")
    print(f"Using model: {model_path if model_path else 'default'}")
    subprocess.run(cmd, check=True)

def run_model_training(iteration, train_data_path, val_data_path, training_mode="dpo", tag="", model_path=None, 
                      use_weighted_dpo=False, weight_scale=1.0, stability_threshold=1.0,
                      batch_size=1, gradient_accumulation_steps=8, learning_rate=1e-5,
                      num_epochs=3, warmup_steps=100, weight_decay=0.01):
    """Run the model training for current iteration"""
    output_dir = f"checkpoints_iteration{iteration}" + (f"_{tag}" if tag else "")
    
    # Determine which model to use for training
    if model_path:
        base_model = model_path
    else:
        # Use the HuggingFace format model from previous iteration
        base_model = f"checkpoints_iteration{iteration-1}/hf_model" if iteration > 0 else "AI4PD/ZymCTRL"
    
    cmd = [
        "python", "lightning_trainer.py",
        "--model_name", base_model,
        "--val_data", val_data_path,
        "--train_data", train_data_path,
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--num_epochs", str(num_epochs),
        "--warmup_steps", str(warmup_steps),
        "--weight_decay", str(weight_decay),
        "--training_mode", training_mode,
        "--output_dir", output_dir,
        "--tag", tag,
        "--stability_threshold", str(stability_threshold)
    ]
    
    # Add weighted DPO args if using that mode
    if training_mode == "dpo" and use_weighted_dpo:
        cmd.extend([
            "--use_weighted_dpo",
            "--weight_scale", str(weight_scale)
        ])
    
    print(f"\n=== Training model for iteration {iteration} ===")
    print(f"Training mode: {training_mode}" + (" (weighted)" if use_weighted_dpo else ""))
    print(f"Using model: {base_model}")
    print(f"Training config:")
    print(f"- Batch size: {batch_size}")
    print(f"- Gradient accumulation: {gradient_accumulation_steps}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {num_epochs}")
    print(f"- Warmup steps: {warmup_steps}")
    print(f"- Weight decay: {weight_decay}")
    print(f"- Stability threshold: {stability_threshold}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run iterative training loop for ZymCTRL')
    # Basic parameters
    parser.add_argument('--start_iteration', type=int, default=0,
                      help='Starting iteration number (default: 0)')
    parser.add_argument('--n_iterations', type=int, default=5,
                      help='Number of iterations to run (default: 5)')
    parser.add_argument('--sequences_per_iteration', type=int, default=100,
                      help='Number of sequences to generate per iteration')
    parser.add_argument('--ec_label', type=str, default="4.2.1.1",
                      help='EC number for target enzyme (default: 4.2.1.1)')
    
    # Training parameters
    parser.add_argument('--training_mode', type=str, default="dpo",
                      choices=['sft', 'dpo'],
                      help='Training mode: sft or dpo (default: dpo)')
    parser.add_argument('--use_weighted_dpo', action='store_true',
                      help='Use weighted DPO with stability score differences as weights')
    parser.add_argument('--weight_scale', type=float, default=1.0,
                      help='Scaling factor for stability difference weights in weighted DPO')
    parser.add_argument('--stability_threshold', type=float, default=1.0,
                      help='Minimum stability score difference for DPO pairs')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Training batch size (default: 1)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                      help='Number of gradient accumulation steps (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate (default: 1e-5)')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs per iteration (default: 3)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                      help='Number of warmup steps (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay for AdamW optimizer (default: 0.01)')
    
    # Model and checkpoint parameters
    parser.add_argument('--initial_model', type=str, default="",
                      help='Optional: Path to initial model checkpoint to start from')
    parser.add_argument('--checkpoint_freq', type=int, default=1,
                      help='Save checkpoint every N iterations (default: 1)')
    
    # Experiment tracking
    parser.add_argument('--tag', type=str, default="",
                      help='Optional tag for experiment tracking')
    
    args = parser.parse_args()
    
    print(f"\n=== Starting Iterative Training ===")
    print(f"Configuration:")
    print(f"- Starting from iteration: {args.start_iteration}")
    print(f"- Running for {args.n_iterations} iterations")
    print(f"- Generating {args.sequences_per_iteration} sequences per iteration")
    print(f"- Training mode: {args.training_mode}" + (" (weighted)" if args.use_weighted_dpo else ""))
    print(f"- Stability threshold: {args.stability_threshold}")
    print(f"\nTraining hyperparameters:")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Epochs: {args.num_epochs}")
    print(f"- Warmup steps: {args.warmup_steps}")
    print(f"- Weight decay: {args.weight_decay}")
    if args.use_weighted_dpo:
        print(f"- Weight scale: {args.weight_scale}")
    print(f"- Tag: {args.tag if args.tag else 'None'}")
    if args.initial_model:
        print(f"- Starting from custom model: {args.initial_model}")
    print("\n")
    
    current_model = args.initial_model
    for i in range(args.start_iteration, args.start_iteration + args.n_iterations):
        print(f"\n=== Starting Iteration {i} ===")
        
        # Generate new data using current model
        # run_data_generation(i, args.ec_label, args.sequences_per_iteration, args.tag, current_model)

        # Get path to generated data
        data_dir = f"training_data_iteration{i}" + (f"_{args.tag}" if args.tag else "")
        train_data = os.path.join(data_dir, "sequences.csv")

        # Get validation data
        val_data_dir = "val_data"
        val_data = os.path.join(val_data_dir, "sequences.csv")
        
        # Train model on new data
        run_model_training(
            iteration=i,
            train_data_path=train_data,
            val_data_path=val_data,
            training_mode=args.training_mode,
            tag=args.tag,
            model_path=current_model,
            use_weighted_dpo=args.use_weighted_dpo,
            weight_scale=args.weight_scale,
            stability_threshold=args.stability_threshold,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay
        )
        
        # Update current model path for next iteration if using checkpoints
        if args.checkpoint_freq > 0 and (i + 1) % args.checkpoint_freq == 0:
            checkpoint_dir = f"checkpoints_iteration{i}" + (f"_{args.tag}" if args.tag else "")
            current_model = os.path.join(checkpoint_dir, "hf_model")  # Use the HF model directory
        
        print(f"\n=== Completed Iteration {i} ===")
        
    print("\nIterative training completed!")

if __name__ == "__main__":
    main() 