# Adversarial Reasoning at Jailbreaking Time - Reproduction

This is a reproduction of the paper "Adversarial Reasoning at Jailbreaking Time" (arXiv:2502.01633v2) adapted to use a local Qwen 4B model and custom data.

## Overview

The code implements the adversarial reasoning algorithm described in the paper, which uses:
- **Attacker LLM**: Generates jailbreaking prompts
- **Feedback LLM**: Analyzes prompts and provides feedback
- **Refiner LLM**: Refines reasoning strings based on feedback
- **GWW Buffer**: Maintains top-performing reasoning strings

## Setup

### Prerequisites

```bash
pip install torch numpy
```

### Directory Structure

```
adversarial_reasoning_reproduction/
├── main.py                      # Main entry point
├── adversarial_reasoning.py     # Core algorithm (Algorithm 1)
├── buffer.py                    # GWW buffer implementation
├── model_wrapper.py             # Local model wrapper
├── prompts.py                   # System prompts
├── utils.py                     # Utility functions
├── data/                        # Data directory
│   └── JBB.csv                 # Dataset (symlink or copy)
└── results/                     # Output directory
    └── results.jsonl           # Results in JSONL format
```

## Usage

### Basic Usage

```bash
python main.py --csv ./data/JBB.csv
```

### Advanced Options

```bash
python main.py \
    --csv ./data/JBB.csv \
    --model_path /path/to/qwen_open_4B.py \
    --num_iterations 15 \
    --num_prompts 16 \
    --num_branches 8 \
    --buffer_size 32 \
    --output_dir ./results \
    --start_idx 0 \
    --end_idx 10
```

### Parameters

- `--csv`: Path to JBB dataset CSV file (default: `./data/JBB.csv`)
- `--model_path`: Path to local Qwen model file (default: specified in code)
- `--num_iterations`: Number of iterations T (default: 15)
- `--num_prompts`: Number of attacking prompts n per iteration (default: 16)
- `--num_branches`: Number of feedback branches m (default: 8)
- `--buffer_size`: Buffer size B for GWW algorithm (default: 32)
- `--batch_divs`: Batch divisions k for feedback (default: 2)
- `--output_dir`: Output directory for results (default: `./results`)
- `--start_idx`: Start index in dataset (default: 0)
- `--end_idx`: End index in dataset (default: None, processes all)

## Algorithm Overview

The implementation follows Algorithm 1 from the paper:

1. **Initialize**: Start with root reasoning string S^(0)
2. **Iterate T times**:
   - Select best reasoning string from buffer
   - Generate n attacking prompts using Attacker LLM
   - Compute losses for each prompt
   - Generate m feedbacks comparing prompts
   - Refine reasoning string using Refiner LLM
   - Add new candidates to buffer
3. **Output**: Best jailbreaking prompt found

## Key Components

### Attacker LLM
Generates creative jailbreaking prompts based on reasoning instructions. Returns JSON with:
- `"Thoughts"`: Reasoning about the approach
- `"Prompt P"`: The actual jailbreaking prompt

### Feedback LLM
Analyzes sorted prompts (by loss) and provides feedback on what makes some more effective. Returns JSON with:
- `"Pattern_observed"`: Common patterns
- `"Comparisons"`: Detailed prompt comparisons
- `"Final_feedback"`: Actionable feedback

### Refiner LLM
Incorporates feedback into the reasoning string. Returns JSON with:
- `"Feedback_points"`: Key points to incorporate
- `"Improved_variable"`: Updated reasoning string

### Loss Function
The algorithm uses cross-entropy loss of the target string as defined in Equation (3.2) of the paper:
```
L_T(P, y_I) = -Σ log P_T(y_i | [P, y_{1:i-1}])
```

For the local model without logit access, we use a proxy loss based on target string matching.

## Results

Results are saved in JSONL format to `{output_dir}/results.jsonl` with the following fields:
- `task_idx`: Task index
- `goal`: Jailbreaking goal
- `target`: Target string
- `category`: Behavior category
- `success`: Whether jailbreak succeeded
- `best_prompt`: Best attacking prompt found
- `best_response`: Target model's response
- `min_loss`: Minimum loss achieved
- `iterations`: Number of iterations run

## Differences from Original

1. **No wandb**: Removed wandb logging as requested
2. **Local Model**: Uses local Qwen 4B model instead of API-based models
3. **Simplified Loss**: Uses proxy loss function since we don't have direct logit access
4. **Single Model**: Uses same model for all LLM roles (Attacker, Feedback, Refiner)

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{sabbaghi2025adversarial,
  title={Adversarial Reasoning at Jailbreaking Time},
  author={Sabbaghi, Mahdi and Kassianik, Paul and Pappas, George and Singer, Yaron and Karbasi, Amin and Hassani, Hamed},
  journal={arXiv preprint arXiv:2502.01633},
  year={2025}
}
```

## License

Please refer to the original repository for license information.
