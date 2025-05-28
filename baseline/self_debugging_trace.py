"""
Self-Debugging (Trace) Baseline

This baseline collects feedback through step-by-step execution traces showing 
intermediate variable values and program execution flow. This variant follows 
the trace-based debugging approach described in Chen et al. (Self-Debugging).
"""

import json
from loguru import logger
from typing import List, Dict, Any
from common import get_completion_with_retry, extract_fixed_code, collect_execution_traces


def ut_trace_baseline(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> str:
    """
    Self-Debugging (Trace) baseline that asks for step-by-step execution tracing.
    
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
    The following Python code is buggy. Please trace through the execution step by step, identify the issue, and provide the corrected version.
    Function name: `{entry_point}`
    
    Buggy code:
    {buggy_code}
    
    Test cases:
    {json.dumps(test_cases, indent=2)}
    
    Execution traces:
    {execution_traces_str}
    
    Please:
    1. Trace through the code execution step by step
    2. Identify where the bug occurs
    3. Provide the fixed code in a Python code block (```python ... ```)
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
        logger.error(f"Error in Self-Debugging (Trace) baseline: {str(e)}")
        return None 