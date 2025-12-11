#!/usr/bin/env python3
"""
Test GPU configuration
"""

import torch
import sys
import os

print("="*80)
print("GPU DETECTION TEST")
print("="*80)

# Test 1: Check PyTorch CUDA
print("\n[1] PyTorch CUDA Support:")
print(f"    CUDA available: {torch.cuda.is_available()}")
print(f"    CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"    PyTorch version: {torch.__version__}")

# Test 2: List all GPUs
if torch.cuda.is_available():
    print(f"\n[2] Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"        Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"        Compute: {props.major}.{props.minor}")
else:
    print(f"\n[2] No GPUs detected")

# Test 3: Test CUDA_VISIBLE_DEVICES
print("\n[3] Testing CUDA_VISIBLE_DEVICES:")

test_configs = [
    ("0", "Single GPU 0"),
    ("1", "Single GPU 1"),
    ("0,1", "Multiple GPUs 0,1"),
    ("3,4", "Multiple GPUs 3,4"),
]

for gpu_ids, desc in test_configs:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    # Need to reimport torch to apply changes (or restart Python)
    print(f"    {desc} (CUDA_VISIBLE_DEVICES={gpu_ids})")
    print(f"        Would make physical GPU {gpu_ids} visible")

# Test 4: Memory test
print("\n[4] GPU Memory Test:")
if torch.cuda.is_available():
    try:
        # Allocate a small tensor
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        print(f"    ✓ Successfully allocated tensor on {device}")
        print(f"    Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        del x
        torch.cuda.empty_cache()
        print(f"    ✓ Memory freed")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
else:
    print("    Skipped (no CUDA)")

print("\n" + "="*80)
print("USAGE EXAMPLES:")
print("="*80)
print("\n# Use GPU 0:")
print("  python main.py --csv ./data/JBB.csv --gpu 0")
print("\n# Use GPU 3:")
print("  python main.py --csv ./data/JBB.csv --gpu 3")
print("\n# Use GPUs 3,4:")
print("  python main.py --csv ./data/JBB.csv --gpu 3,4")
print("\n# Use CPU (default):")
print("  python main.py --csv ./data/JBB.csv")
print("\n" + "="*80)
