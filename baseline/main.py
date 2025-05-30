#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from loguru import logger
from typing import List, Dict, Any

# Import baseline methods
from no_debugging import repeat_generation
from simple_feedback import simple_baseline
from self_edit import ut_baseline
from self_debugging_explanation import ut_expl_baseline
from self_debugging_trace import ut_trace_baseline

def load_test_cases(dataset_path: str) -> List[Dict[str, Any]]:
    """Load test cases from the dataset file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def setup_logging(output_dir: str):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    logger.add(os.path.join(output_dir, "all_info.log"))

def run_baseline(baseline_type: str, buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]] = None) -> str:
    """Run a specific baseline method."""
    if baseline_type == "no_debugging":
        return repeat_generation(buggy_code, entry_point)
    elif baseline_type == "simple_feedback":
        return simple_baseline(buggy_code, entry_point)
    elif baseline_type == "self_edit":
        return ut_baseline(buggy_code, entry_point, test_cases)
    elif baseline_type == "self_debugging_explanation":
        return ut_expl_baseline(buggy_code, entry_point, test_cases)
    elif baseline_type == "self_debugging_trace":
        return ut_trace_baseline(buggy_code, entry_point, test_cases)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

def evaluate_baselines(dataset_path: str, output_dir: str):
    """Evaluate all baseline methods on the given dataset."""
    # Load test cases
    test_cases = load_test_cases(dataset_path)
    
    # Setup logging
    setup_logging(output_dir)
    
    # List of baseline methods to evaluate
    baselines = [
        "no_debugging",
        "simple_feedback",
        "self_edit",
        "self_debugging_explanation",
        "self_debugging_trace"
    ]
    
    # Results dictionary to store statistics
    results = {}
    
    # Evaluate each baseline
    for baseline in baselines:
        logger.info(f"Running {baseline} baseline...")
        success_count = 0
        total_count = len(test_cases)
        
        # Process each test case
        for test_case in tqdm(test_cases, desc=f"Processing {baseline}"):
            try:
                buggy_code = test_case["buggy_code"]
                entry_point = test_case["entry_point"]
                test_data = test_case.get("test_cases")
                
                # Run baseline method
                fixed_code = run_baseline(baseline, buggy_code, entry_point, test_data)
                
                # Verify if the fix was successful (this would depend on your verification logic)
                if fixed_code and fixed_code != buggy_code:
                    success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing test case with {baseline}: {str(e)}")
        
        # Calculate success rate
        success_rate = (success_count / total_count) * 100
        results[baseline] = {
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": success_rate
        }
        
        # Log results
        logger.info(f"{baseline} Results:")
        logger.info(f"Success Rate: {success_rate:.2f}%")
        logger.info(f"Successful Fixes: {success_count}/{total_count}")
        
    # Save results to file
    results_file = os.path.join(output_dir, "baseline_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Run and evaluate baseline debugging methods")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save results")
    args = parser.parse_args()
    
    evaluate_baselines(args.dataset, args.output_dir)

if __name__ == "__main__":
    main() 