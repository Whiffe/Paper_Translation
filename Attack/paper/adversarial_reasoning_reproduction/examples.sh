#!/bin/bash
# Example commands for running adversarial reasoning with different GPU configurations

echo "======================================"
echo "Adversarial Reasoning - GPU Examples"
echo "======================================"
echo ""

# Example 1: Quick test with GPU 0
echo "Example 1: Quick test with GPU 0 (1 task, 5 iterations)"
echo "Command:"
echo "  python main.py --csv ./data/JBB.csv --gpu 0 --start_idx 0 --end_idx 1 --num_iterations 5"
echo ""

# Example 2: Use GPU 3 for 10 tasks
echo "Example 2: Use GPU 3 for 10 tasks"
echo "Command:"
echo "  python main.py --csv ./data/JBB.csv --gpu 3 --start_idx 0 --end_idx 10"
echo ""

# Example 3: Use GPUs 3,4 for full run
echo "Example 3: Use GPUs 3,4 for full dataset"
echo "Command:"
echo "  python main.py --csv ./data/JBB.csv --gpu 3,4 --num_iterations 15"
echo ""

# Example 4: CPU mode
echo "Example 4: CPU mode (slower, no GPU required)"
echo "Command:"
echo "  python main.py --csv ./data/JBB.csv --start_idx 0 --end_idx 1"
echo ""

# Example 5: Parallel execution
echo "Example 5: Parallel execution on multiple GPUs"
echo "Commands (run in separate terminals):"
echo "  Terminal 1: python main.py --csv ./data/JBB.csv --gpu 0 --start_idx 0 --end_idx 25 &"
echo "  Terminal 2: python main.py --csv ./data/JBB.csv --gpu 1 --start_idx 25 --end_idx 50 &"
echo "  Terminal 3: python main.py --csv ./data/JBB.csv --gpu 2 --start_idx 50 --end_idx 75 &"
echo "  Terminal 4: python main.py --csv ./data/JBB.csv --gpu 3 --start_idx 75 --end_idx 100 &"
echo ""

echo "======================================"
echo "To run an example, copy the command and execute it"
echo "======================================"
