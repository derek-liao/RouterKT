import wandb
import yaml
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import time
import argparse
import hashlib
import datetime

def generate_sweep_name(data_name):
    """Generate a unique sweep name using dataset name and hash"""
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a hash using timestamp
    hash_input = f"{data_name}_{timestamp}"
    hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Combine dataset name and hash
    sweep_name = f"{data_name}_{hash_id}"
    return sweep_name

def run_agent(sweep_id, gpu_id, sweep_configuration, num_gpus):
    """Run a sweep agent on a specific GPU"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Calculate number of runs per GPU
    total_combinations = (
        len(sweep_configuration["parameters"]["routing_mode"]["values"]) *
        len(sweep_configuration["parameters"]["num_selected_heads"]["values"]) *
        len(sweep_configuration["parameters"]["balance_loss_weight"]["values"]) *
        len(sweep_configuration["parameters"]["l2"]["values"])
    )
    runs_per_gpu = total_combinations // num_gpus + (1 if total_combinations % num_gpus > gpu_id else 0)
    
    print(f"Starting agent on GPU {gpu_id}, will run {runs_per_gpu} experiments")
    
    # Start the agent process
    process = subprocess.Popen([
        "wandb", "agent", sweep_id,
        "--count", str(runs_per_gpu)
    ], env=env)
    
    return process

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to train"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        required=True,
        help="The name of the dataset to use in training"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for the sweep"
    )
    args = parser.parse_args()

    sweep_configuration = {
        "program": "main.py",
        "method": "grid",
        "metric": {
            "name": "auc_d",
            "goal": "maximize"
        },
        "parameters": {
            "routing_mode": {
                "values": ["dynamic"]
            },
            "num_selected_heads": {
                "values": [1, 2, 4]
            },
            "num_shared_heads": {
                "values": [1, 2, 4]
            },
            "balance_loss_weight": {
                "values": [0, 0.001, 0.01]
            },
            "l2": {
                "values": [0, 0.0001, 0.001, 0.01]
            },
            "model_name": {
                "value": args.model_name
            },
            "data_name": {
                "value": args.data_name
            },
            "use_wandb": {
                "value": 1
            },
            "disable_visualization": {
                "value": 1
            },
            "gpu_num": {
                "value": 0  # This will be overridden by CUDA_VISIBLE_DEVICES
            }
        }
    }

    # Generate sweep name
    sweep_name = generate_sweep_name(args.data_name)
    
    # Initialize sweep with the generated name
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project="RouterKT-sweep"
    )
    print(f"Created sweep with ID: {sweep_id} and name: {sweep_name}")

    # Start agents on specified number of GPUs
    processes = []
    for gpu_id in range(args.num_gpus):
        process = run_agent(sweep_id, gpu_id, sweep_configuration, args.num_gpus)
        processes.append(process)
        time.sleep(2)  # Small delay between agent starts to avoid conflicts
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()

if __name__ == "__main__":
    main()