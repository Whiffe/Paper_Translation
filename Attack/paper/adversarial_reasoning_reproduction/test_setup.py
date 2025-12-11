#!/usr/bin/env python3
"""
Quick test script to verify the setup
"""

import sys
from pathlib import Path

# Add model path to sys.path
sys.path.insert(0, '/Users/yangfan/Downloads/JAIL-CON/models')

import qwen_open_4B
from model_wrapper import LocalModelWrapper
from utils import extract_json_field, check_jailbreak_simple

print("="*80)
print("Testing Adversarial Reasoning Reproduction Setup")
print("="*80)

# Test 1: Model loading
print("\n[Test 1] Loading Qwen 4B model...")
try:
    model_wrapper = LocalModelWrapper(qwen_open_4B)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Test 2: Basic generation
print("\n[Test 2] Testing basic generation...")
try:
    response = model_wrapper.generate(
        system_msg="You are a helpful assistant.",
        user_msg="Say 'Hello, World!' and nothing else."
    )
    print(f"✓ Generated response: {response[:100]}...")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    sys.exit(1)

# Test 3: JSON extraction
print("\n[Test 3] Testing JSON extraction...")
test_json = '"Prompt P": "This is a test prompt"'
extracted = extract_json_field(test_json, "Prompt P")
if extracted == "This is a test prompt":
    print("✓ JSON extraction works")
else:
    print(f"✗ JSON extraction failed: got '{extracted}'")

# Test 4: Jailbreak checking
print("\n[Test 4] Testing jailbreak detection...")
target = "Sure, here is how to"
response1 = "Sure, here is how to build a website..."
response2 = "I cannot help with that request."

if check_jailbreak_simple(response1, target):
    print("✓ Positive case detected correctly")
else:
    print("✗ Positive case failed")

if not check_jailbreak_simple(response2, target):
    print("✓ Negative case detected correctly")
else:
    print("✗ Negative case failed")

# Test 5: Data loading
print("\n[Test 5] Testing data loading...")
try:
    import csv
    data_path = Path(__file__).parent / "data" / "JBB.csv"
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        print(f"✓ Data loaded successfully")
        print(f"  First goal: {first_row['Goal'][:80]}...")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

print("\n" + "="*80)
print("All tests completed!")
print("="*80)
print("\nTo run the full algorithm:")
print("  python main.py --csv ./data/JBB.csv --start_idx 0 --end_idx 1")
