import os
import argparse
import subprocess
from pathlib import Path

"""
WORK IN PROGRESS: THIS WILL BECOME THE MAIN TRAINING SCRIPT. RUN ONLINE TRAINING

"""

def run_data_generation(iteration, ec_label, n_batches, tag="", model_path=None):
    """Run the data generation script for current iteration"""
    cmd = [
        "python", "generate_training_data.py",
        "--iteration_num", str(iteration),
        "--ec_label", ec_label,
        "--n_batches", str(n_batches),
        "--tag", tag
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])
    print(f"\n=== Generating data for iteration {iteration} ===")
    subprocess.run(cmd, check=True)

def run_model_training(iteration, train_data_path, training_mode="dpo", tag="", model_path=None):
    """Run the model training for current iteration"""
    output_dir = f"checkpoints_iteration{iteration}" + (f"_{tag}" if tag else "")
    
    # Determine which model to use for training
    if model_path:
        base_model = model_path
    else:
        base_model = f"output_iteration{iteration-1}" if iteration > 0 else "AI4PD/ZymCTRL"
    
    cmd = [
        "python", "lightning_trainer.py",
        "--model_name", base_model,
        "--train_data", train_data_path,
        "--batch_size", "4",
        "--learning_rate", "1e-5",
        "--num_epochs", "3",
        "--training_mode", training_mode,
        "--output_dir", output_dir,
        "--tag", tag
    ]
    print(f"\n=== Training model for iteration {iteration} ===")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run iterative training loop for ZymCTRL')
    parser.add_argument('--start_iteration', type=int, default=0,
                      help='Starting iteration number (default: 0)')
    parser.add_argument('--n_iterations', type=int, default=5,
                      help='Number of iterations to run (default: 5)')
    parser.add_argument('--ec_label', type=str, default="4.2.1.1",
                      help='EC number for target enzyme (default: 4.2.1.1)')
    parser.add_argument('--sequences_per_iteration', type=int, default=100,
                      help='Number of sequences to generate per iteration')
    parser.add_argument('--training_mode', type=str, default="dpo",
                      choices=['sft', 'dpo'],
                      help='Training mode: sft or dpo (default: dpo)')
    parser.add_argument('--tag', type=str, default="",
                      help='Optional tag for experiment tracking')
    parser.add_argument('--initial_model', type=str, default="",
                      help='Optional: Path to initial model checkpoint to start from')
    parser.add_argument('--checkpoint_freq', type=int, default=1,
                      help='Save checkpoint every N iterations (default: 1)')
    
    args = parser.parse_args()
    
    # Calculate batches needed (assuming 2 sequences per batch from generate_training_data.py)
    n_batches = args.sequences_per_iteration // 2
    
    print(f"Starting iterative training loop:")
    print(f"- Starting from iteration: {args.start_iteration}")
    print(f"- Running for {args.n_iterations} iterations")
    print(f"- Generating {args.sequences_per_iteration} sequences per iteration")
    print(f"- Training mode: {args.training_mode}")
    print(f"- Tag: {args.tag if args.tag else 'None'}")
    if args.initial_model:
        print(f"- Starting from custom model: {args.initial_model}")
    
    current_model = args.initial_model
    for i in range(args.start_iteration, args.start_iteration + args.n_iterations):
        print(f"\n=== Starting Iteration {i} ===")
        
        # Generate new data using current model
        run_data_generation(i, args.ec_label, n_batches, args.tag, current_model)
        
        # Get path to generated data
        data_dir = f"training_data_iteration{i}" + (f"_{args.tag}" if args.tag else "")
        train_data = os.path.join(data_dir, "sequences.csv")
        
        # Train model on new data
        run_model_training(i, train_data, args.training_mode, args.tag, current_model)
        
        # Update current model path for next iteration if using checkpoints
        if args.checkpoint_freq > 0 and (i + 1) % args.checkpoint_freq == 0:
            checkpoint_dir = f"checkpoints_iteration{i}" + (f"_{args.tag}" if args.tag else "")
            current_model = checkpoint_dir  # Will use latest checkpoint from this directory
        
        print(f"\n=== Completed Iteration {i} ===")
        
    print("\nIterative training completed!")

if __name__ == "__main__":
    main() 