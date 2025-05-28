"""
Simple Feedback Baseline

This baseline collects minimal feedback by providing only the buggy code 
and basic error information to the model.
"""

from loguru import logger
from common import get_completion_with_retry, extract_fixed_code


def simple_baseline(buggy_code: str, entry_point: str) -> str:
    """
    Simple feedback baseline that provides only buggy code to the model.
    
    Args:
        buggy_code: The buggy code to fix
        entry_point: Function entry point name
        
    Returns:
        str: Fixed code or None if failed
    """
    prompt = f"""
    The following Python code is buggy. Please fix it and provide the corrected version.
    Function name: `{entry_point}`
    
    Buggy code:
    {buggy_code}
    
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
        logger.error(f"Error in simple baseline: {str(e)}")
        return None 