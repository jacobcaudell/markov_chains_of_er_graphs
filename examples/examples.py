#!/usr/bin/env python3
"""
Example usage of the Markov Chain of ER Graphs package.
Run this script to see various use cases and expected outputs.
"""

import subprocess
import sys
import os

def run_example(name, command):
    """Run an example command and display results."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {name}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Failed to run example: {e}")

def main():
    print("Markov Chain of ER Graphs - Example Usage")
    print("=" * 50)
    
    # Example 1: Basic simulation
    run_example(
        "Basic Monte Carlo Simulation",
        "python greedy_reorder_mc.py --n 30 --T 6 --trials 5 --quiet"
    )
    
    # Example 2: Different distance modes
    run_example(
        "Hamming Distance Mode",
        "python greedy_reorder_mc.py --n 30 --T 6 --trials 5 --dist hamming --quiet"
    )
    
    # Example 3: Debug analysis
    run_example(
        "Single Trial Debug Analysis",
        "python debug_trial.py --n 30 --T 6 --dist noisy --eps 0.2"
    )
    
    # Example 4: Compare all modes
    run_example(
        "Compare All Distance Modes",
        "python compare_modes.py --n 30 --T 6 --trials 5 --quiet"
    )
    
    print(f"\n{'='*60}")
    print("All examples completed!")
    print("Check the generated JSON files for detailed results.")
    print("Use 'python analyze_debug.py <file.json>' to view results.")

if __name__ == "__main__":
    main()
