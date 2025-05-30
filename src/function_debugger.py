import time
from loguru import logger
import json
from llm_client import get_completion_with_retry
from utils import extract_code_blocks, extract_function
from config import MODEL, MAX_PARSE_RETRIES, RETRY_DELAY
import config


def debug_function(function_code, function_name, test_cases, max_parse_retries=MAX_PARSE_RETRIES):
    """Debug a single function using test cases"""
    config.TOTAL_DEBUG_FUNCTION_CALLS += 1

    logger.info(f"Debugging function:\n{function_code}\nWith test cases: {test_cases}")
    prompt = f"""
    Debug the following Python function. The function is not passing all test cases.
    Analyze the code, identify the bug, and provide a fixed version of the function.

    ### Function:
    {function_code}

    ### Test Cases:
    {json.dumps(test_cases, indent=2)}
    
    ### Task:

    Provide your response in the following format:
    1. try to work as a python interpreter to execute the code step-by-step.
    2. identify the change of each variable as you "run" the code line-by-line.
    3. based on the execution trace, try to identify the bug
    4. provide the final fixed code in a Python code block (```python ... ```)

    Make sure to include the entire function, including the function signature. And you will be rewarded to simulate the code execution in your mind and provide step-by-step trace of the code execution.
    
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant helping to debug Python functions.'},
        {'role': 'user', 'content': prompt},
    ]

    if "starcoder2" in MODEL.lower():
        # remove the system message
        messages = messages[1:]

    for attempt in range(max_parse_retries):
        try:
            response = get_completion_with_retry(messages)
            code_blocks = extract_code_blocks(response)
            if code_blocks:
                # search in a reverse order to find the final fixed function
                for block in code_blocks[::-1]:
                    fixed_function = extract_function(block, function_name)
                    if fixed_function:
                        analysis = response.split("```python")[0].strip()
                        logger.info("Generated debug analysis and fixed code")
                        return analysis, fixed_function
            raise ValueError("No valid fixed function found in the response")
        except ValueError as e:
            logger.error(f"Failed to extract fixed function (attempt {attempt + 1}/{max_parse_retries}): {str(e)}")
            if attempt < max_parse_retries - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Returning None for analysis and fixed code.")
                return None, None 