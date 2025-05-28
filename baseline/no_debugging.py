"""
No-Debugging Baseline (Repeat Generation)

This baseline generates seed buggy solutions for other debugging baselines to repair.
We follow the codes from LDB to set up this baseline.
"""

from loguru import logger
from common import get_completion_with_retry, extract_fixed_code


def repeat_generation(problem_prompt: str, entry_point: str) -> str:
    """
    No-debugging baseline that simply regenerates code from the problem prompt.
    
    Args:
        problem_prompt: The original problem description
        entry_point: Function entry point name
        
    Returns:
        str: Generated code or None if failed
    """
    prompt = f"""
    Problem: {problem_prompt}
    
    Please provide Python code in a Python code block (```python ... ```) to solve the problem above.
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in generating Python code.'},
        {'role': 'user', 'content': prompt},
    ]

    try:
        logger.info(f"Prompt: {prompt}")
        response = get_completion_with_retry(messages)
        logger.info(f"Response: {response}")
        fixed_function = extract_fixed_code(response)
        return fixed_function
    except Exception as e:
        logger.error(f"Error in no-debugging baseline: {str(e)}")
        return None 