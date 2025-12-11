#!/bin/bash
# Quick start script

echo "==================================="
echo "Adversarial Reasoning Reproduction"
echo "==================================="
echo ""

# Check if data exists
if [ ! -f "data/JBB.csv" ]; then
    echo "Error: data/JBB.csv not found!"
    echo "Please create a symlink or copy your dataset:"
    echo "  ln -s /path/to/your/JBB.csv data/JBB.csv"
    exit 1
fi

# Run test
echo "Running setup test..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup test passed!"
    echo ""
    echo "Example commands:"
    echo "  # Run on first task only (quick test)"
    echo "  python main.py --csv ./data/JBB.csv --start_idx 0 --end_idx 1"
    echo ""
    echo "  # Run on first 10 tasks"
    echo "  python main.py --csv ./data/JBB.csv --start_idx 0 --end_idx 10"
    echo ""
    echo "  # Run with fewer iterations (faster)"
    echo "  python main.py --csv ./data/JBB.csv --num_iterations 5 --start_idx 0 --end_idx 1"
else
    echo ""
    echo "Setup test failed. Please check the errors above."
    exit 1
fi
