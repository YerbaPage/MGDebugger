import traceback
import json
from loguru import logger
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
from utils import split_nested_functions, get_dependency_graph_str, evaluate, parse_json_response, extract_code_blocks, extract_functions, extract_function, create_dependency_graph, topological_sort, merge_changes_to_parents, evaluate_simple, parse_transcoder_problem_content
from test_parser import get_parameter_names, parse_tests
import os
import time
import sys
from typing import List, Dict, Any

MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# OpenAI client setup
client = OpenAI(
    base_url="http://localhost:18889/v1",
    api_key="token-abc123s",
)

# Parameters
MAX_OUTER_RETRY = 10
CONTINUOUS_RETRY = True
TEMPERATURE = 0

# Stats
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_DEBUG_CALLS = 0


def get_completion_with_retry(messages, model=MODEL, max_retries=3):
    """
    Get completion from LLM with retry mechanism.
    
    Args:
        messages: List of message dictionaries for the chat completion
        model: Model name to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        str: The completion response content
    """
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_DEBUG_CALLS
    TOTAL_DEBUG_CALLS += 1
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=TEMPERATURE
            )
            TOTAL_PROMPT_TOKENS += chat_completion.usage.prompt_tokens
            TOTAL_COMPLETION_TOKENS += chat_completion.usage.completion_tokens
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
            else:
                logger.error("Max retries reached. Giving up.")
                raise


def collect_execution_traces(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> List[str]:
    """
    Collect execution traces for test cases on buggy code.
    
    Args:
        buggy_code: The buggy code to test
        entry_point: Function entry point name
        test_cases: List of test case dictionaries
        
    Returns:
        List[str]: List of execution trace strings
    """
    execution_traces = []
    for i, test_case in enumerate(test_cases):
        result, trace = evaluate(buggy_code, entry_point, test_case, return_trace=True)
        if result:
            execution_traces.append(f"Test case {i + 1}: Passed")
        else:
            execution_traces.append(f"Test case {i + 1}:\nInput: {test_case['input']}\nExpected: {test_case['expected_output']}\nActual: {trace['actual_output']}")
            if 'traceback' in trace:
                execution_traces.append(f"Traceback: {trace['traceback']}")
    return execution_traces


def extract_fixed_code(response: str) -> str:
    """
    Extract fixed code from LLM response.
    
    Args:
        response: LLM response containing code blocks
        
    Returns:
        str: Extracted fixed code or None if not found
    """
    code_blocks = extract_code_blocks(response)
    if code_blocks:
        # take the last code block as the fixed function
        return code_blocks[-1]
    raise ValueError("No valid fixed function found in the response")