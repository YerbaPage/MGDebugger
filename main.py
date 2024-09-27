import time
from openai import OpenAI
from collections import Counter
from loguru import logger
import json
import os
from tqdm import tqdm
import sys
import io
import traceback
import ast
import time
import re
from groq import Groq
from utils import split_nested_functions, get_dependency_graph_str, evaluate, parse_json_response, extract_code_blocks, extract_functions, extract_function, create_dependency_graph, topological_sort, merge_changes_to_parents, evaluate_simple, parse_transcoder_problem_content
from test_parser import get_parameter_names, parse_tests

# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code --dtype auto --api-key token-abc123s --port 18889 --max-model-len 40960 
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/CodeQwen1.5-7B-Chat --dtype auto --api-key token-abc123s --port 18892 --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.5
# CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --model TechxGenus/Codestral-22B-v0.1-GPTQ --dtype auto --api-key token-abc123s --port 18890 --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.5 --chat-template helper/codestral_template.jinja
MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# MODEL = "TechxGenus/Codestral-22B-v0.1-GPTQ"
# MODEL = "Qwen/CodeQwen1.5-7B-Chat"


client = OpenAI(
    base_url="http://localhost:18889/v1",
    api_key="token-abc123s",
)

# dscoder 18889
# codestral 18890
# codeqwen 18892

# Hyperparameters
MAX_VLLM_RETRIES = 10  # maximum number of retries for the VLLM call
MAX_PARSE_RETRIES = 3  # we sometimes fail to parse the code/test cases from the response, so we retry
MAX_DEBUG_RETRIES = 3  # 3 seems to be slightly better than 1
RETRY_DELAY = 0  # seconds
REPEAT_CONVERT_HIERARCHICAL_NUM = 1  # seems unimportant
REPEAT_TEST_CASE_GENERATION_NUM = 1  # 1 seems to be better than 3
MAX_OUTER_RETRY = 10  # maximum number of retries for the entire debugging process
CONTINUOUS_RETRY = False  # whether to retry from the last fixed code, else retry from the original buggy code. it seems better to set this to False
TEMPERATURE = 0.8  # 0.8 better than 1.0 better than 0.2
MINP = 0.05

# for collecting stats
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_MG_DEBUG_CALLS = 0
TOTAL_DEBUG_FUNCTION_CALLS = 0
TOTAL_GENERATE_TEST_CASES_CALLS = 0
TOTAL_CONVERT_HIERARCHICAL_CALLS = 0


def get_completion_with_retry(messages, model=MODEL, MAX_VLLM_RETRIES=MAX_VLLM_RETRIES):
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
    for attempt in range(MAX_VLLM_RETRIES):
        try:
            logger.info(f"Attempting LLM call (attempt {attempt + 1}/{MAX_VLLM_RETRIES})")
            logger.info(f"Input messages: {messages[-1]['content']}")
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=TEMPERATURE,
                extra_body={"min_p": MINP}
            )
            response = chat_completion.choices[0].message.content
            logger.info(f"LLM response: {response}")
            logger.info("LLM call received")

            # Update token counts
            TOTAL_PROMPT_TOKENS += chat_completion.usage.prompt_tokens
            TOTAL_COMPLETION_TOKENS += chat_completion.usage.completion_tokens

            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            if attempt < MAX_VLLM_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Giving up.")
                raise


def generate_test_cases(full_code, gold_test_cases, function_name, MAX_PARSE_RETRIES=MAX_PARSE_RETRIES):
    global TOTAL_GENERATE_TEST_CASES_CALLS
    TOTAL_GENERATE_TEST_CASES_CALLS += 1

    logger.info(f"Generating test cases for function: {function_name}")

    # TODO: replace the var1 with ...
    # TODO: compare the performance with/without the indent before the prompt (now it has an indent)
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

    # TODO: the model still likes to generate the tests before analyzing them case by case

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


def debug_function(function_code, function_name, test_cases, MAX_PARSE_RETRIES=MAX_PARSE_RETRIES):
    global TOTAL_DEBUG_FUNCTION_CALLS
    TOTAL_DEBUG_FUNCTION_CALLS += 1

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

    for attempt in range(MAX_PARSE_RETRIES):
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
            logger.error(f"Failed to extract fixed function (attempt {attempt + 1}/{MAX_PARSE_RETRIES}): {str(e)}")
            if attempt < MAX_PARSE_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Returning None for analysis and fixed code.")
                return None, None


def convert_to_hierarchical(code, include_example=False):
    global TOTAL_CONVERT_HIERARCHICAL_CALLS
    TOTAL_CONVERT_HIERARCHICAL_CALLS += 1

    logger.info("Converting code to hierarchical structure")

    example = """
    ### Example of tree-style hierarchical structure:
    
    ```python
    def main_function(input):
        preprocessed_data = preprocess(input)
        result = process(preprocessed_data)
        return result

    def preprocess(data):
        cleaned_data = clean_data(data)
        normalized_data = normalize_data(cleaned_data)
        return normalized_data

    def clean_data(data):
        # Implementation of data cleaning
        pass

    def normalize_data(data):
        # Implementation of data normalization
        pass

    def process(data):
        feature_vector = extract_features(data)
        result = classify(feature_vector)
        return result

    def extract_features(data):
        # Implementation of feature extraction
        pass

    def classify(feature_vector):
        # Implementation of classification
        pass
    ```
    """ if include_example else ""

    prompt = f"""
    Convert the following Python code into a tree-style hierarchical structure with multiple levels of sub-functions.
    Each significant step or logical block should be its own function, and functions can call other sub-functions.
    Ensure that the main function calls these sub-functions in the correct order, creating a tree-like structure.

    ### Original Code:
    {code}

    {example}

    ### Instructions:
    Please first analyze the codes step by step, and then provide the converted code in a Python code block (```python ... ```). When providing the final converted code, make sure to include all the functions in a flattened format, where each function is defined separately.
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in refactoring Python code into a tree-style hierarchical structure.'},
        {'role': 'user', 'content': prompt},
    ]

    if "starcoder2" in MODEL.lower():
        
        # remove the system message
        messages = messages[1:]

    best_conversion = None
    max_subfunctions = 0

    # TODO: remove unused functions
    # TODO: bugs exist in building the dependency graph, we need to use ast instead of string matching to determine dependencies in some cases where functions have similar names. Though it won't hurt the performance much, it will increase unnecessary time cost.

    for _ in range(REPEAT_CONVERT_HIERARCHICAL_NUM):
        response = get_completion_with_retry(messages)
        code_blocks = extract_code_blocks(response)

        if code_blocks:
            converted_code = code_blocks[0]
            subfunctions = len(extract_functions(converted_code)) - 1  # Subtract 1 to exclude the main function

            if subfunctions > max_subfunctions:
                max_subfunctions = subfunctions
                best_conversion = converted_code

    if best_conversion:
        logger.info(f"Converted code to tree-style hierarchical structure with {max_subfunctions} sub-functions")
        # split nested functions
        best_conversion = split_nested_functions(best_conversion)
        return best_conversion
    else:
        logger.error("Failed to convert code to tree-style hierarchical structure")
        code = split_nested_functions(code)
        return code


def mg_debug(full_code, gold_test_cases, max_debug_attempts=MAX_DEBUG_RETRIES):
    global TOTAL_MG_DEBUG_CALLS
    TOTAL_MG_DEBUG_CALLS += 1
    logger.info("Starting main debugging process")

    convert_hierarchical_attempts = 0
    while convert_hierarchical_attempts < MAX_PARSE_RETRIES:
        try:
            # Convert to tree-style hierarchical structure
            hierarchical_code = convert_to_hierarchical(full_code, include_example=False)
            logger.info(f"Converted code to tree-style hierarchical structure:\n{hierarchical_code}")

            functions = extract_functions(hierarchical_code)
            # TODO: remove unused functions

            # Create a dependency graph
            dependency_graph = create_dependency_graph(functions)
            logger.info(f"Dependency graph:\n{get_dependency_graph_str(dependency_graph)}")
            
            break
        except Exception as e:
            logger.error(f"Failed to convert code to hierarchical structure (attempt {convert_hierarchical_attempts + 1}/{MAX_PARSE_RETRIES}): {str(e)}")
            convert_hierarchical_attempts += 1
            # retry

    # Sort functions based on their dependencies (bottom-up)
    sorted_functions = topological_sort(dependency_graph)
    logger.info(f"Sorted functions: {sorted_functions}")

    for func_name in sorted_functions:
        logger.info(f"Processing function: {func_name}")

        func_code = functions[func_name]
        test_cases = generate_test_cases(hierarchical_code, gold_test_cases, func_name)
        fixed_code = func_code

        for debug_attempt in range(max_debug_attempts):
            all_tests_pass = True

            for test_case in test_cases:
                passed, result = evaluate(hierarchical_code, func_name, test_case)
                if not passed:
                    all_tests_pass = False
                    break

            if all_tests_pass:
                logger.info(f"All tests passed for function: {func_name}")
                break

            logger.info(f"Debugging function: {func_name} (Attempt {debug_attempt + 1}/{max_debug_attempts})")
            analysis, new_fixed_code = debug_function(fixed_code, func_name, test_cases)

            if new_fixed_code:
                logger.info(f"New fixed code for {func_name}:\n{new_fixed_code}")
                fixed_code = new_fixed_code
                functions[func_name] = fixed_code
                logger.info(f"Merging {func_name} changes")
                hierarchical_code = merge_changes_to_parents(func_name, dependency_graph, functions)
                logger.info(f"Code after merging updates in {func_name}:\n{hierarchical_code}")
            else:
                logger.warning(f"Failed to fix {func_name}. Keeping previous implementation.")
                break

        if not all_tests_pass:
            logger.warning(f"Could not fix {func_name} after {max_debug_attempts} attempts. Keeping original implementation.")

    # Reconstruct the full code with fixed functions
    fixed_full_code = "\n\n".join(functions.values())
    logger.info("Debugging process completed. Reconstructed full code.")

    return fixed_full_code


def test():
    buggy_code = '''
def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with the supplied string. """
    
    def is_palindrome(s: str) -> bool:
        return s == s[::-1]

    suffix_start = 0
    for i in range(len(string)):
        if is_palindrome(string[i:]):
            suffix_start = i
            
    return string + string[:suffix_start][::-1]
    '''.strip()

    gold_test_cases = [
        {'input': 'cat', 'expected_output': 'catac'},
        {'input': 'cata', 'expected_output': 'catac'},
        {'input': '', 'expected_output': ''}
    ]

    entry_point = 'make_palindrome'

    fixed_code = mg_debug(buggy_code, gold_test_cases)
    logger.info(f"Fixed code:\n{fixed_code}")

    logger.info("============= Final evaluation with private test cases =============")
    # evaluate the final codes with private testcases
    all_tests = gold_test_cases
    for testcase in all_tests:
        result, testcase = evaluate(fixed_code, entry_point, testcase)
        logger.info(f"Passed: {result}, Test case: {testcase}")


def debug_humaneval(input_seeds: str, max_examples: int = None, output_folder: str = None):
    fixed_problems = 0
    total_unsolved = 0

    with open(input_seeds, "r") as f:
        seeds = f.readlines()

    unsolved_seeds = []
    # filter those problems that are not solved
    for i in range(len(seeds)):
        problem = json.loads(seeds[i])
        if not problem["is_solved"]:
            unsolved_seeds.append(problem)

    # resume_index = 95
    # unsolved_seeds = unsolved_seeds[106:]
    # unsolved_seeds = unsolved_seeds[:resume_index]
    
    # resume_path = "output_data/mbpp/seed/codeqwen/20240909-025624/CodeQwen1.5-7B-Chat_debugging_seeds_from_codeqwen.jsonl"
    # # only load "debugged": false problems from the resume path
    # with open(resume_path, "r") as f:
    #     resume_problems = f.readlines()
    # resume_unsolved_task_ids = [json.loads(problem)["task_id"] for problem in resume_problems if not json.loads(problem)["debugged"]]
    # unsolved_seeds = [problem for problem in unsolved_seeds if problem["task_id"] in resume_unsolved_task_ids]
    # logger.info(f"Resuming from {resume_path}, {len(unsolved_seeds)} problems to debug")

    # filter the unsolved problems that are not in the resume path

    # parse transcoder problems
    if "transcoder" in input_seeds.lower():
        logger.info(f"Parsing the problem content for transcoder problems")
        unsolved_seeds = [parse_transcoder_problem_content(problem) for problem in tqdm(unsolved_seeds)]

    total_unsolved = len(unsolved_seeds)
    logger.info(f"Debugging {total_unsolved} unsolved problems")
    if max_examples is not None:
        unsolved_seeds = unsolved_seeds[:max_examples]
        logger.info(f"Filtering to {max_examples} examples")

    for problem in tqdm(unsolved_seeds, ncols=100):
        
        model_to_be_fixed = input_seeds.split("/")[-2]
        model_name = MODEL.split("/")[-1]
        with open(f"{output_folder}/{model_name}_debugging_seeds_from_{model_to_be_fixed}.jsonl", "w+") as f:
            for seed in unsolved_seeds:
                f.write(json.dumps(seed) + "\n")
                
        logger.info(f"Processing unsolved problem: {problem['task_id']}")
        logger.info(f"Problem: {problem}")
        logger.info(f"Problem Raw Prompt: \n{problem['prompt']}")

        try:
            buggy_code = problem["solution"]
            entry_point = problem["entry_point"]
            try:
                parameter_names = get_parameter_names(problem["prompt"], entry_point)
            except:
                parameter_names = get_parameter_names(problem["solution"], entry_point)
            logger.info(f"Extracted parameter names: {parameter_names}")

            # in order to save time, we extract the first 3 given tests for transcoder
            if "transcoder" in problem["task_id"].lower():
                logger.info(f"Extracted {len(problem['given_tests'])} given tests, only using the first 3 samples")
                problem["given_tests"] = problem["given_tests"][:3]

            gold_tests_raw = "\n".join(problem["given_tests"]).replace(entry_point, "candidate")
            gold_tests = parse_tests(gold_tests_raw, parameter_names, entry_point)["test_cases"]
            logger.info(f"Extracted gold test cases: {gold_tests}")

            problem['fixed_codes'] = []  # Initialize list to store fixed codes for each retry
            problem['mg_debug_retries'] = 0  # Initialize retry counter

            for outer_retry in range(MAX_OUTER_RETRY):
                fixed_code = mg_debug(buggy_code, gold_tests)
                problem['fixed_codes'].append(fixed_code)
                problem['mg_debug_retries'] += 1

                all_passed = evaluate_simple(fixed_code, entry_point, problem["test"])
                if all_passed:
                    fixed_problems += 1
                    problem['debugged'] = True
                    logger.info(f"Successfully fixed problem: {problem['task_id']} on retry {outer_retry + 1}")
                    break
                else:
                    logger.info(f"Failed to fix problem: {problem['task_id']} on retry {outer_retry + 1}")
                    if outer_retry < MAX_OUTER_RETRY - 1:
                        if CONTINUOUS_RETRY:
                            buggy_code = fixed_code  # Use the last fixed code as the new buggy code for the next retry
                        else:
                            # Use the original buggy code for the next retry
                            buggy_code = problem["solution"]

            if not all_passed:
                problem['debugged'] = False
                logger.info(f"Failed to fix problem: {problem['task_id']} after {MAX_OUTER_RETRY} retries")

        except Exception as e:
            logger.error(f"Error occurred while processing problem: {problem['task_id']}")
            logger.error(traceback.format_exc())
            problem['fixed_codes'] = []
            problem['debugged'] = False
            problem['mg_debug_retries'] = 0

    # Add statistics to the log
    logger.info("=== Statistics ===")
    logger.info(f"Total prompt tokens: {TOTAL_PROMPT_TOKENS}")
    logger.info(f"Total completion tokens: {TOTAL_COMPLETION_TOKENS}")
    logger.info(f"Total mg_debug calls: {TOTAL_MG_DEBUG_CALLS}")
    logger.info(f"Total debug_function calls: {TOTAL_DEBUG_FUNCTION_CALLS}")
    logger.info(f"Total generate_test_cases calls: {TOTAL_GENERATE_TEST_CASES_CALLS}")
    logger.info(f"Total convert_hierarchical calls: {TOTAL_CONVERT_HIERARCHICAL_CALLS}")

    # Compute and log average statistics
    avg_prompt_tokens = TOTAL_PROMPT_TOKENS / TOTAL_MG_DEBUG_CALLS if TOTAL_MG_DEBUG_CALLS > 0 else 0
    avg_completion_tokens = TOTAL_COMPLETION_TOKENS / TOTAL_MG_DEBUG_CALLS if TOTAL_MG_DEBUG_CALLS > 0 else 0
    avg_debug_function_calls = TOTAL_DEBUG_FUNCTION_CALLS / TOTAL_MG_DEBUG_CALLS if TOTAL_MG_DEBUG_CALLS > 0 else 0
    avg_generate_test_cases_calls = TOTAL_GENERATE_TEST_CASES_CALLS / TOTAL_MG_DEBUG_CALLS if TOTAL_MG_DEBUG_CALLS > 0 else 0

    logger.info(f"Average prompt tokens per mg_debug call: {avg_prompt_tokens:.2f}")
    logger.info(f"Average completion tokens per mg_debug call: {avg_completion_tokens:.2f}")
    logger.info(f"Average debug_function calls per mg_debug call: {avg_debug_function_calls:.2f}")
    logger.info(f"Average generate_test_cases calls per mg_debug call: {avg_generate_test_cases_calls:.2f}")

    # distribution of debug retries that solved the problem
    debug_retries = [problem['mg_debug_retries'] for problem in unsolved_seeds if problem['debugged']]
    debug_retries_counter = Counter(debug_retries)

    logger.info(f"=== Final Results ===")
    logger.info(f"Total unsolved problems in seeds: {total_unsolved}")
    logger.info(f"Problems fixed by our method: {fixed_problems}")
    logger.info(f"Success rate: {fixed_problems / total_unsolved * 100:.2f}%")
    logger.info(f"Debug retries distribution for solved problems: {debug_retries_counter}")

    logger.info(f"Total solved problems before: {len(seeds) - total_unsolved}")
    logger.info(f"Total solved problems after: {len(seeds) - total_unsolved + fixed_problems}")
    logger.info(f"Previous accuracy: {(len(seeds) - total_unsolved) / len(seeds) * 100:.2f}%")
    logger.info(f"Final accuracy: {(len(seeds) - total_unsolved + fixed_problems) / len(seeds) * 100:.2f}%")

    # save the final results to a file
    model_to_be_fixed = input_seeds.split("/")[-2]
    model_name = MODEL.split("/")[-1]
    with open(f"{output_folder}/{model_name}_debugging_seeds_from_{model_to_be_fixed}.jsonl", "w+") as f:
        for seed in unsolved_seeds:
            f.write(json.dumps(seed) + "\n")

    with open(f"{output_folder}/statistics.json", "w") as f:
        stats = {
            "total_prompt_tokens": TOTAL_PROMPT_TOKENS,
            "total_completion_tokens": TOTAL_COMPLETION_TOKENS,
            "total_mg_debug_calls": TOTAL_MG_DEBUG_CALLS,
            "total_debug_function_calls": TOTAL_DEBUG_FUNCTION_CALLS,
            "total_generate_test_cases_calls": TOTAL_GENERATE_TEST_CASES_CALLS,
            "total_convert_hierarchical_calls": TOTAL_CONVERT_HIERARCHICAL_CALLS,
            "avg_prompt_tokens_per_mg_debug": avg_prompt_tokens,
            "avg_completion_tokens_per_mg_debug": avg_completion_tokens,
            "avg_debug_function_calls_per_mg_debug": avg_debug_function_calls,
            "avg_generate_test_cases_calls_per_mg_debug": avg_generate_test_cases_calls
        }
        json.dump(stats, f, indent=2)
        scores = {
            "total_unsolved": total_unsolved,
            "fixed_problems": fixed_problems,
            "success_rate": fixed_problems / total_unsolved * 100,
            "debug_retries_distribution": debug_retries_counter,
            "total_solved_problems_before": len(seeds) - total_unsolved,
            "total_solved_problems_after": len(seeds) - total_unsolved + fixed_problems,
            "previous_accuracy": (len(seeds) - total_unsolved) / len(seeds) * 100,
            "final_accuracy": (len(seeds) - total_unsolved + fixed_problems) / len(seeds) * 100
        }
        json.dump(scores, f, indent=2)


if __name__ == "__main__":

    input_seeds = "input_data/humaneval/seed/deepseekcoder/seed.jsonl"
    # input_seeds = "input_data/humaneval/seed/codestral/seed.jsonl"
    # input_seeds = "input_data/humaneval/seed/codeqwen/seed.jsonl"

    # input_seeds = "input_data/mbpp/seed/codestral/seed.jsonl"
    # input_seeds = "input_data/mbpp/seed/deepseekcoder/seed.jsonl"
    # input_seeds = "input_data/mbpp/seed/codeqwen/seed.jsonl"
    
    # input_seeds = "input_data/humanevalfix/seeds.jsonl"

    seed_stamp = input_seeds.split("input_data/")[-1].replace("/seed.jsonl", "")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = f"./output_data/{seed_stamp}/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(f"{output_folder}/all_info.log", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    # save all important params in a file
    with open(f"{output_folder}/params.json", "w+") as f:
        info = {
            "input_seeds": input_seeds,
            "output_folder": output_folder,
            "MAX_DEBUG_RETRIES": MAX_DEBUG_RETRIES,
            "MAX_PARSE_RETRIES": MAX_PARSE_RETRIES,
            "RETRY_DELAY": RETRY_DELAY,
            "REPEAT_CONVERT_HIERARCHICAL_NUM": REPEAT_CONVERT_HIERARCHICAL_NUM,
            "REPEAT_TEST_CASE_GENERATION_NUM": REPEAT_TEST_CASE_GENERATION_NUM,
            "MODEL": MODEL,
            "MAX_OUTER_RETRY": MAX_OUTER_RETRY,
            "CONTINUOUS_RETRY": CONTINUOUS_RETRY,
            "TEMPERATURE": TEMPERATURE,
            "MAX_VLLM_RETRIES": MAX_VLLM_RETRIES,
            "MINP": MINP
        }
        f.write(json.dumps(info, indent=2))

    debug_humaneval(input_seeds, max_examples=None, output_folder=output_folder)
