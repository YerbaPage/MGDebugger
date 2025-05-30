from collections import Counter
from loguru import logger
import json
from llm_client import get_completion_with_retry
from utils import parse_json_response
from config import MODEL, MAX_PARSE_RETRIES, REPEAT_TEST_CASE_GENERATION_NUM
import config


def generate_test_cases(full_code, gold_test_cases, function_name, max_parse_retries=MAX_PARSE_RETRIES):
    """Generate test cases for a specific function based on full code and gold test cases"""
    config.TOTAL_GENERATE_TEST_CASES_CALLS += 1

    logger.info(f"Generating test cases for function: {function_name}")

    prompt = f"""
    Analyze the following Python code and focus on the function named `{function_name}`. 
    Generate the same number of test cases for this specific function based on the provided gold test cases for the main function.
    Ensure that the generated test cases are consistent with the behavior expected in the gold test cases.

    ### Full Code
    {full_code}

    ### Gold Test Cases for the main function
    {json.dumps(gold_test_cases, indent=2)}

    ### Function to generate test cases for: `{function_name}`

    ### Output format:
    **Test Case 1:**
    Input: ...
    Analysis: ...
    Expected Output: ...
    
    **Test Case 2:**
    ...
    
    **All Test Cases:**
    ```json
    {{
        "test_cases": [
            {{"input": {{'var1': 'value1', 'var2': 'value2'}}, "expected_output": "expected_output"}},
            ...
        ]
    }}
    ```
    
    ### Hint
    Analyze how the `{function_name}` function is used within the main function and how it contributes to the expected outputs in the gold test cases. Generate test cases that reflect this behavior. For each test case, you should analyze step-by-step based on both the input and the expected output of the main function, and then provide the corresponding input and expected output for the `{function_name}` function.
    """

    messages = [
        {'role': 'system', 'content': "You are an AI assistant specialized in analyzing Python functions and generating test cases."},
        {'role': 'user', 'content': prompt},
    ]

    if "starcoder2" in MODEL.lower():
        # remove the system message
        messages = messages[1:]

    all_test_cases = []
    for _ in range(REPEAT_TEST_CASE_GENERATION_NUM):
        response = get_completion_with_retry(messages)
        parsed_response = parse_json_response(response)
        if parsed_response and 'test_cases' in parsed_response:
            all_test_cases.extend(parsed_response['test_cases'])

    if not all_test_cases:
        logger.error("Failed to generate any valid test cases")
        return []

    # Perform majority voting on expected outputs
    logger.info(f"Generated {len(all_test_cases)} test cases for {function_name}: {all_test_cases}")
    test_case_counter = Counter([json.dumps(tc) for tc in all_test_cases])
    # show the frequency of each test case
    logger.info(f"Test case frequency:")
    for tc, freq in test_case_counter.items():
        logger.info(f"Test case: {tc}, Frequency: {freq}")
    most_common_test_cases = test_case_counter.most_common(len(gold_test_cases))
    final_test_cases = [json.loads(tc[0]) for tc in most_common_test_cases]

    logger.info(f"Voted {len(final_test_cases)} test cases for {function_name}: {final_test_cases}")
    return final_test_cases 