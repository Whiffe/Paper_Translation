#!/usr/bin/env python3
"""
Adversarial Reasoning at Jailbreaking Time - Reproduction
Based on the paper: arxiv:2502.01633v2
Adapted to use local Qwen model and custom data
"""

import argparse
import csv
import json
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Import local modules
from buffer import GWW_dfs_min
from model_wrapper import LocalModelWrapper
from adversarial_reasoning import AdversarialReasoning


def load_dataset(csv_path):
    """Load JBB dataset from CSV file"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'goal': row['Goal'],
                'target': row['Target'],
                'behavior': row['Behavior'],
                'category': row['Category'],
                'source': row['Source']
            })
    return data


def setup_gpu(gpu_ids):
    """
    Setup GPU devices for PyTorch

    Args:
        gpu_ids: String like "0", "0,1", "3,4" or None for CPU
    """
    if gpu_ids is None:
        print("Running on CPU")
        return 'cpu'

    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    # Parse GPU IDs
    gpu_list = [int(x.strip()) for x in gpu_ids.split(',')]

    # Check CUDA availability
    if not torch.cuda.is_available():
        print(f"WARNING: CUDA not available, falling back to CPU")
        return 'cpu'

    # Get actual number of visible GPUs after setting CUDA_VISIBLE_DEVICES
    num_gpus = torch.cuda.device_count()

    print(f"Using GPU(s): {gpu_ids}")
    print(f"Number of visible GPUs: {num_gpus}")

    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Return primary device
    return f'cuda:0'  # After setting CUDA_VISIBLE_DEVICES, always use cuda:0


def main():
    parser = argparse.ArgumentParser(description='Adversarial Reasoning Reproduction')
    parser.add_argument('--csv', type=str, default='./data/JBB.csv',
                        help='Path to JBB dataset CSV file')
    parser.add_argument('--model_path', type=str,
                        default='/Users/yangfan/Downloads/JAIL-CON/models/qwen_open_4B.py',
                        help='Path to local model file')
    parser.add_argument('--num_iterations', type=int, default=15,
                        help='Number of iterations per task (T in paper)')
    parser.add_argument('--num_prompts', type=int, default=16,
                        help='Number of attacking prompts per iteration (n in paper)')
    parser.add_argument('--num_branches', type=int, default=8,
                        help='Number of feedback branches (m in paper)')
    parser.add_argument('--buffer_size', type=int, default=32,
                        help='Buffer size for GWW algorithm (B in paper)')
    parser.add_argument('--batch_divs', type=int, default=2,
                        help='Number of divisions for feedback batching (k in paper)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index in dataset')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index in dataset (None for all)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device(s) to use (e.g., "0", "0,1", "3,4"). None for CPU')

    args = parser.parse_args()

    # Setup GPU
    print("\n" + "="*80)
    print("GPU CONFIGURATION")
    print("="*80)
    device = setup_gpu(args.gpu)
    print("="*80 + "\n")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.csv}...")
    dataset = load_dataset(args.csv)

    # Select subset
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset = dataset[args.start_idx:end_idx]
    print(f"Loaded {len(dataset)} tasks")

    # Import model module dynamically
    print(f"Loading model from {args.model_path}...")
    sys.path.insert(0, str(Path(args.model_path).parent))
    import qwen_open_4B

    # Initialize model wrapper (device will be handled by model internally)
    model_wrapper = LocalModelWrapper(qwen_open_4B, device=device)

    # Initialize adversarial reasoning algorithm
    algorithm = AdversarialReasoning(
        model_wrapper=model_wrapper,
        num_iterations=args.num_iterations,
        num_prompts=args.num_prompts,
        num_branches=args.num_branches,
        buffer_size=args.buffer_size,
        batch_divs=args.batch_divs
    )

    # Run algorithm on each task
    results = []
    for idx, task in enumerate(dataset):
        print(f"\n{'='*80}")
        print(f"Task {idx+1}/{len(dataset)}: {task['goal'][:100]}...")
        print(f"{'='*80}\n")

        result = algorithm.run(
            goal=task['goal'],
            target=task['target'],
            task_idx=idx
        )

        results.append({
            'task_idx': idx + args.start_idx,
            'goal': task['goal'],
            'target': task['target'],
            'category': task['category'],
            'success': result['success'],
            'best_prompt': result['best_prompt'],
            'best_response': result['best_response'],
            'min_loss': float(result['min_loss']),
            'iterations': result['iterations']
        })

        # Save intermediate results
        output_file = Path(args.output_dir) / 'results.jsonl'
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + '\n')

        print(f"\nTask {idx+1} completed. Success: {result['success']}")
        print(f"Min loss: {result['min_loss']:.4f}")

    # Calculate and print summary statistics
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(results)}")
    print(f"Successful jailbreaks: {sum(r['success'] for r in results)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Results saved to: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
