"""
Self-Debugging (Explanation) Baseline

This baseline collects comprehensive feedback including complete unit test execution 
results, detailed execution traces, and error messages. We implement this baseline 
following the approach described in Chen et al. (Self-Debugging), implementing the 
generation, explanation, and feedback steps, using unit tests to produce feedback 
and leveraging LLMs for explanation.
"""

import json
from loguru import logger
from typing import List, Dict, Any
from common import get_completion_with_retry, extract_fixed_code, collect_execution_traces


def ut_expl_baseline(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> str:
    """
    Self-Debugging (Explanation) baseline that asks for step-by-step explanation.
    
    Args:
        buggy_code: The buggy code to fix
        entry_point: Function entry point name
        test_cases: List of test case dictionaries
        
    Returns:
        str: Fixed code or None if failed
    """
    execution_traces = collect_execution_traces(buggy_code, entry_point, test_cases)
    execution_traces_str = "\n\n".join(execution_traces)

    prompt = f"""
    The following Python code is buggy. Please analyze the code step by step, explain what might be wrong, and then provide the corrected version.
    Function name: `{entry_point}`
    
    Buggy code:
    {buggy_code}
    
    Test cases:
    {json.dumps(test_cases, indent=2)}
    
    Unit test feedback:
    {execution_traces_str}
    
    Please:
    1. First, explain what you think is wrong with the code
    2. Then provide the fixed code in a Python code block (```python ... ```)
    Make sure to include the entire function, including the function signature.
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in debugging Python code.'},
        {'role': 'user', 'content': prompt},
    ]

    try:
        logger.info(f"Prompt: {prompt}")
        response = get_completion_with_retry(messages)
        logger.info(f"Response: {response}")
        fixed_function = extract_fixed_code(response)
        return fixed_function
    except Exception as e:
        logger.error(f"Error in Self-Debugging (Explanation) baseline: {str(e)}")
        return None
