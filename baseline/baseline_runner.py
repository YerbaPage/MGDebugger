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

# MODEL = "codellama/CodeLlama-13b-Instruct-hf"
# MODEL = "codellama/CodeLlama-34b-Instruct-hf"
# MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
# MODEL = "deepseek-ai/deepseek-coder-33b-instruct"
MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# MODEL = "TechxGenus/Codestral-22B-v0.1-GPTQ"
# MODEL = "mistralai/Codestral-22B-v0.1"
# MODEL = "Qwen/CodeQwen1.5-7B-Chat"
# MODEL = "bigcode/starcoder2-15b-instruct-v0.1"

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/CodeQwen1.5-7B-Chat --dtype auto --api-key token-abc123s --port 18892 --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.5

# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model bigcode/starcoder2-15b-instruct-v0.1 --dtype auto --api-key token-abc123s --port 18891 --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.6


# OpenAI client setup
client = OpenAI(
    base_url="http://localhost:18889/v1",
    api_key="token-abc123s",
)

# dscoder 18889
# codestral 18890
# starcoder 18891
# codeqwen 18892


# BASELINE = "repeat"
BASELINE = "simple"
# BASELINE = "ut" # Self-Evolve
# BASELINE = "ut_expl"
# BASELINE = "ut_trace"

# Parameters
MAX_OUTER_RETRY = 10
CONTINUOUS_RETRY = True
TEMPERATURE = 0

# Stats
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_DEBUG_CALLS = 0


def get_completion_with_retry(messages, model=MODEL, max_retries=3):
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


def repeat_generation(problem_prompt: str, entry_point: str) -> str:
    prompt = f"""
    Problem: {problem_prompt}
    
    Please provide Pythons code in a Python code block (```python ... ```) to solve the problem above.
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in generating Python code.'},
        {'role': 'user', 'content': prompt},
    ]

    try:
        logger.info(f"Prompt: {prompt}")
        response = get_completion_with_retry(messages)
        logger.info(f"Response: {response}")
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            # take the last code block as the fixed function
            fixed_function = code_blocks[-1]
            return fixed_function
        raise ValueError("No valid fixed function found in the response")
    except Exception as e:
        logger.error(f"Error in simple baseline: {str(e)}")
        return None


def simple_baseline(buggy_code: str, entry_point: str) -> str:
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
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            # take the last code block as the fixed function
            fixed_function = code_blocks[-1]
            return fixed_function
        raise ValueError("No valid fixed function found in the response")
    except Exception as e:
        logger.error(f"Error in simple baseline: {str(e)}")
        return None


def collect_execution_traces(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> List[str]:
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


def ut_baseline(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> str:
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
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            # take the last code block as the fixed function
            fixed_function = code_blocks[-1]
            return fixed_function
        raise ValueError("No valid fixed function found in the response")
    except Exception as e:
        logger.error(f"Error in UT baseline: {str(e)}")
        return None


def ut_expl_baseline(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> str:
    execution_traces = collect_execution_traces(buggy_code, entry_point, test_cases)
    execution_traces_str = "\n\n".join(execution_traces)

    prompt = f"""
    The following Python code is buggy. Please fix it based on the provided test cases and feedback.
    
    Buggy code:
    {buggy_code}
    
    Test cases:
    {json.dumps(test_cases, indent=2)}
    
    Unit test feedback:
    {execution_traces_str}
    
    Please firstly explain the functionality of the original code line by line, and then correct the code based on the unit test feedback. Remember to wrap the corrected code in a Python code block (```python ... ```).
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in debugging Python code.'},
        {'role': 'user', 'content': prompt},
    ]

    try:
        logger.info(f"Prompt: {prompt}")
        response = get_completion_with_retry(messages)
        logger.info(f"Response: {response}")
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            # take the last code block as the fixed function
            fixed_function = code_blocks[-1]
            return fixed_function
        raise ValueError("No valid fixed function found in the response")
    except Exception as e:
        logger.error(f"Error in UT baseline: {str(e)}")
        return None


def ut_trace_baseline(buggy_code: str, entry_point: str, test_cases: List[Dict[str, Any]]) -> str:
    execution_traces = collect_execution_traces(buggy_code, entry_point, test_cases)
    execution_traces_str = "\n\n".join(execution_traces)

    prompt = f"""
    The following Python code is buggy. Please fix it based on the provided test cases and feedback.
    
    Buggy code:
    {buggy_code}
    
    Test cases:
    {json.dumps(test_cases, indent=2)}
    
    Unit test feedback:
    {execution_traces_str}
    
    Please firstly trace through the execution of the code (i.e. trace the change of variables after each operation) to determine what needs to be fixed, and correct the codes. Remember to wrap the corrected code in a Python code block (```python ... ```).
    """

    messages = [
        {'role': 'system', 'content': 'You are an AI assistant specialized in debugging Python code.'},
        {'role': 'user', 'content': prompt},
    ]

    try:
        logger.info(f"Prompt: {prompt}")
        response = get_completion_with_retry(messages)
        logger.info(f"Response: {response}")
        code_blocks = extract_code_blocks(response)
        if code_blocks:
            # take the last code block as the fixed function
            fixed_function = code_blocks[-1]
            return fixed_function
        raise ValueError("No valid fixed function found in the response")
    except Exception as e:
        logger.error(f"Error in UT baseline: {str(e)}")
        return None


def debug_with_baseline(input_seeds, baseline_type, max_examples=None, output_folder=None):
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

    # # only select 5 problems for debugging
    # unsolved_seeds = unsolved_seeds[132:]
    
    # only keep the one with problem_id = "HumanEval/123"
    unsolved_seeds = [problem for problem in unsolved_seeds if problem["task_id"] == "HumanEval/123"]

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
            problem['debug_retries'] = 0  # Initialize retry counter

            for retry in range(MAX_OUTER_RETRY):
                if baseline_type == 'simple':
                    fixed_code = simple_baseline(buggy_code, entry_point)
                elif baseline_type == 'ut':
                    fixed_code = ut_baseline(buggy_code, entry_point, gold_tests)
                elif baseline_type == 'ut_expl':
                    fixed_code = ut_expl_baseline(buggy_code, entry_point, gold_tests)
                elif baseline_type == 'ut_trace':
                    fixed_code = ut_trace_baseline(buggy_code, entry_point, gold_tests)
                elif baseline_type == 'repeat':
                    fixed_code = repeat_generation(problem["prompt"], entry_point)
                else:
                    raise ValueError(f"Unknown baseline type: {baseline_type}")

                problem['fixed_codes'].append(fixed_code)
                problem['debug_retries'] += 1

                if fixed_code:
                    all_passed = evaluate_simple(fixed_code, entry_point, problem["test"])
                    if all_passed:
                        fixed_problems += 1
                        problem['debugged'] = True
                        logger.info(f"Successfully fixed problem: {problem['task_id']} on retry {retry + 1}")
                        # save the log
                        with open(f"{output_folder}/{baseline_type}_baseline_unsolved.jsonl", "w+") as f:
                            for seed in unsolved_seeds:
                                f.write(json.dumps(seed) + "\n")
                        break
                    else:
                        if CONTINUOUS_RETRY:
                            buggy_code = fixed_code
                        else:
                            # Use the original buggy code for the next retry
                            buggy_code = problem["solution"]
                else:
                    logger.warning(f"Failed to get fixed code for {problem['task_id']} on retry {retry + 1}")
                    with open(f"{output_folder}/{baseline_type}_baseline_unsolved.jsonl", "w+") as f:
                        for seed in unsolved_seeds:
                            f.write(json.dumps(seed) + "\n")

            if not problem.get('debugged', False):
                problem['debugged'] = False
                logger.info(f"Failed to fix problem: {problem['task_id']} after {MAX_OUTER_RETRY} retries")

        except Exception as e:
            logger.error(f"Error occurred while processing problem: {problem['task_id']}")
            logger.error(traceback.format_exc())
            problem['debugged'] = False
            
            # save the log
            with open(f"{output_folder}/{baseline_type}_baseline_unsolved.jsonl", "w+") as f:
                for seed in unsolved_seeds:
                    f.write(json.dumps(seed) + "\n")

    # distribution of debug retries that solved the problem
    debug_retries = [problem['debug_retries'] for problem in unsolved_seeds if problem['debugged']]
    debug_retries_counter = Counter(debug_retries)

    # Log results
    logger.info(f"=== {baseline_type.upper()} Baseline Results ===")
    logger.info(f"Total unsolved problems: {total_unsolved}")
    logger.info(f"Problems fixed: {fixed_problems}")
    logger.info(f"Success rate: {fixed_problems / total_unsolved * 100:.2f}%")
    logger.info(f"Retry distribution: {debug_retries_counter}")
    logger.info(f"Total prompt tokens: {TOTAL_PROMPT_TOKENS}")
    logger.info(f"Total completion tokens: {TOTAL_COMPLETION_TOKENS}")
    logger.info(f"Total debug calls: {TOTAL_DEBUG_CALLS}")
    logger.info(f"Average prompt tokens per debug call: {TOTAL_PROMPT_TOKENS / TOTAL_DEBUG_CALLS:.2f}")
    logger.info(f"Average completion tokens per debug call: {TOTAL_COMPLETION_TOKENS / TOTAL_DEBUG_CALLS:.2f}")

    with open(f"{output_folder}/{baseline_type}_baseline_unsolved.jsonl", "w+") as f:
        for seed in unsolved_seeds:
            f.write(json.dumps(seed) + "\n")

    # Save results
    if output_folder:
        with open(f"{output_folder}/{baseline_type}_statistics.json", "w") as f:
            stats = {
                "retry_distribution": debug_retries_counter,
                "total_prompt_tokens": TOTAL_PROMPT_TOKENS,
                "total_completion_tokens": TOTAL_COMPLETION_TOKENS,
                "total_debug_calls": TOTAL_DEBUG_CALLS,
                "avg_prompt_tokens_per_debug": TOTAL_PROMPT_TOKENS / TOTAL_DEBUG_CALLS,
                "avg_completion_tokens_per_debug": TOTAL_COMPLETION_TOKENS / TOTAL_DEBUG_CALLS
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

    return fixed_problems, total_unsolved


if __name__ == "__main__":

    # input_seeds = "ldb/input_data/humaneval/seed/deepseekcoder/seed.jsonl"
    # input_seeds = "ldb/input_data/humaneval/seed/codellama/seed.jsonl"
    # input_seeds = "ldb/input_data/humaneval/seed/codestral/seed.jsonl"
    # input_seeds = "ldb/input_data/humaneval/seed/gpt-4-1106-preview/seed.jsonl"
    # input_seeds = "ldb/input_data/humaneval/seed/reflexion/seed.jsonl"
    # input_seeds = "ldb/input_data/humaneval/seed/codeqwen/seed.jsonl"
    # input_seeds = "ldb/input_data/humaneval/seed/starcoder2/seed.jsonl"

    # input_seeds = "ldb/input_data/mbpp/seed/starcoder/seed.jsonl"
    # input_seeds = "ldb/input_data/mbpp/seed/gpt-3.5-turbo-0613/seed.jsonl"
    # input_seeds = "ldb/input_data/mbpp/seed/codestral/seed.jsonl"
    # input_seeds = "ldb/input_data/mbpp/seed/deepseekcoder/seed.jsonl"
    # input_seeds = "ldb/input_data/mbpp/seed/codeqwen/seed.jsonl"
    # input_seeds = "ldb/input_data/mbpp/seed/starcoder2/seed.jsonl"

    # input_seeds = "ldb/input_data/transcoder/seed/starcoder/seed.jsonl"
    # input_seeds = "ldb/input_data/transcoder/seed/gpt-3.5-turbo-0613/seed.jsonl"
    
    input_seeds = "ldb/input_data/humanevalfix/seeds.jsonl"

    seed_stamp = input_seeds.split("input_data/")[-1].replace("/seed.jsonl", "")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = f"./output_data_baselines/{seed_stamp}/{timestamp}"
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
            "MODEL": MODEL,
            "MAX_OUTER_RETRY": MAX_OUTER_RETRY,
            "CONTINUOUS_RETRY": CONTINUOUS_RETRY,
            "TEMPERATURE": TEMPERATURE,
            "BASELINE": BASELINE
        }
        f.write(json.dumps(info, indent=2))

    fixed_simple, total_unsolved = debug_with_baseline(input_seeds, BASELINE, max_examples=None, output_folder=output_folder)
