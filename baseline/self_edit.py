"""
Self-Edit Baseline

This baseline collects feedback through unit test execution results and 
expected vs. actual outputs for failing test cases. We implement this baseline 
following the methodology described in Zhang et al. (Self-Edit), where we execute 
the generated code on example test cases and embed the results as comments.
"""

import json
from loguru import logger
from typing import List, Dict, Any
from common import get_completion_with_retry, extract_fixed_code, collect_execution_traces


def ut_baseline(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> str:
    """
    Self-Edit baseline that uses unit test execution results for feedback.
    
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
    The following Python code is buggy. Please fix it based on the provided test cases and feedback.
    Function name: `{entry_point}`
    
    Buggy code:
    {buggy_code}
    
    Test cases:
    {json.dumps(test_cases, indent=2)}
    
    Unit test feedback:
    {execution_traces_str}
    
    Please fix the code and provide ONLY the final fixed code in a Python code block (```python ... ```).
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
        logger.error(f"Error in Self-Edit baseline: {str(e)}")
        return None 