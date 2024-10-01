from typing import List, Dict, Any
from loguru import logger
import ast
import re
import json
from tqdm import tqdm


def get_parameter_names(prompt: str, entry_point: str) -> List[str]:
    """
    Extract parameter names from the function signature in the prompt.
    """
    # logger.debug(f"Prompt: {prompt}")
    # logger.debug(f"Entry point: {entry_point}")
    tree = ast.parse(prompt)
    for node in ast.walk(tree):
        # logger.debug(f"Node name: {node.name if hasattr(node, 'name') else None}")
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            # Return the parameter names from the function definition that matches the entry point
            return [param.arg for param in node.args.args]
    return []


def parse_tests(test: str, parameter_names: List[str], entry_point: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse the test string into a structured format using AST.
    """
    # Remove the METADATA section
    test = re.sub(r'METADATA = \{[^}]*\}', '', test)

    # Parse the entire test string
    tree = ast.parse(test)

    test_cases = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            # Process each assert statement
            test_case = process_assert(node, entry_point, parameter_names)
            if test_case:
                test_cases.append(test_case)

    return {"test_cases": test_cases}


def process_assert(node: ast.Assert, entry_point: str, parameter_names: List[str]) -> Dict[str, Any]:
    """
    Process a single assert statement and extract input and expected output.
    """
    if isinstance(node.test, ast.Compare) and isinstance(node.test.ops[0], ast.Eq):
        left = node.test.left
        right = node.test.comparators[0]

        if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "candidate":
            input_dict = process_input(left.args, parameter_names)
            # logger.debug(f"Input: {input_dict}")
            # logger.debug(f"right: {right}")
            # logger.debug(f"right type: {type(right)}")
            # logger.debug(f"right value: {right.name if isinstance(right, ast.Name) else right.s if isinstance(right, ast.Str) else None}")

            try:
                # Attempt to evaluate using literal_eval
                expected_output = ast.literal_eval(right)
            except ValueError:
                # Fallback to eval if literal_eval fails
                # logger.warning("Falling back to eval due to failure in literal_eval")
                expected_output = eval(compile(ast.Expression(right), filename="<ast>", mode="eval"))

            return {"input": input_dict, "expected_output": expected_output}

    return None


def process_input(args: List[ast.expr], parameter_names: List[str]) -> Dict[str, Any]:
    """
    Process the input arguments and match them with parameter names.
    """
    input_dict = {}

    for i, arg in enumerate(args):
        try:
            # Attempt to evaluate using literal_eval for simpler cases
            evaluated_arg = ast.literal_eval(arg)
        except ValueError:
            # Fallback to eval if literal_eval fails
            # logger.warning("Falling back to eval due to failure in literal_eval")
            evaluated_arg = eval(compile(ast.Expression(arg), filename="<ast>", mode="eval"))

        if i < len(parameter_names):
            input_dict[parameter_names[i]] = evaluated_arg
        else:
            # Handle extra arguments if any
            input_dict[f"arg_{i}"] = evaluated_arg

    return input_dict


def parse_all_problems(problems):
    success_count = 0
    unhandled_failures = 0
    for problem in problems:
        try:
            problem = json.loads(problem)

            # logger.info(f"Problem: {problem}")
            # logger.debug(f"Test: {problem['test']}")

            entry_point = problem["entry_point"]
            parameter_names = get_parameter_names(problem["prompt"], entry_point)
            # logger.info(f"Parameter names: {parameter_names}")

            given_tests_raw = "\n".join(problem["given_tests"]).replace(entry_point, "candidate")
            given_tests = parse_tests(given_tests_raw, parameter_names, entry_point)

            # Parse the test cases using the parameter names
            parsed_tests = parse_tests(problem["test"], parameter_names, entry_point)
            # logger.info(f"Parsed tests: {parsed_tests}")
            success_count += 1
        except:
            logger.exception(f"Error processing problem {problem['task_id']}")
            if problem['is_solved'] == False:
                unhandled_failures += 1
            continue

    logger.info(f"Success count: {success_count}")
    logger.info(f"Total problems: {len(problems)}")
    logger.info(f"Unhandled failures: {unhandled_failures}")


def parse_specific_problem(problem):
    try:
        if isinstance(problem, str):
            problem = json.loads(problem)

        logger.info(f"Problem: {problem}")
        logger.debug(f"Test: {problem['test']}")
        logger.debug(f"Given Test: {problem['given_tests']}")

        entry_point = problem["entry_point"]
        parameter_names = get_parameter_names(problem["prompt"], entry_point)
        logger.debug(f"Parameter names: {parameter_names}")

        given_tests_raw = "\n".join(problem["given_tests"]).replace(entry_point, "candidate")
        given_tests = parse_tests(given_tests_raw, parameter_names, entry_point)
        logger.debug(f"Given tests: {given_tests}")

        # Parse the test cases using the parameter names
        all_tests = parse_tests(problem["test"], parameter_names, entry_point)
        logger.debug(f"Parsed tests: {all_tests}")
        return all_tests
    except:
        logger.exception(f"Error processing problem {problem['task_id']}")
        return None

#assert next_smallest([]) is None
#assert decode_cyclic(encode_cyclic("abc")) == "abc"
#assert round(find_zero([-6, 11, -6, 1]), 2) == 1.0
#assert abs(candidate(1.33) - 0.33) < 1e-6

def check_all_problems(problems):
    problems_q = []
    success_count = 0
    fail_count = 0
    for problem in tqdm(problems):
        try:
            problem = json.loads(problem)

            logger.info(f"Problem: {problem}")
            logger.debug(f"Test: {problem['test']}")
            logger.debug(f"All Test: {problem['given_tests']}")

            entry_point = problem["entry_point"]
            parameter_names = get_parameter_names(problem["prompt"], entry_point)
            logger.info(f"Parameter names: {parameter_names}")

            # given_tests_len = len(problem["given_tests"])
            # given_tests_raw = "\n".join(problem["given_tests"]).replace(entry_point, "candidate")
            # given_tests = parse_tests(given_tests_raw, parameter_names, entry_point)
            # parsed_given_tests_len = len(given_tests['test_cases'])
            # assert given_tests_len == parsed_given_tests_len
            # success_count += 1

            #Parse the test cases using the parameter names
            tests_len_candidate =  problem["test"].count('candidate')
            parsed_tests = parse_tests(problem["test"], parameter_names, entry_point)
            parsed_test_len = len(parsed_tests['test_cases'])
            #assert parsed_test_len != 0 
            assert tests_len_candidate - 1 == parsed_test_len
            logger.info(f"Parsed tests: {parsed_tests}")
            success_count += 1
        except:
            logger.exception(f"Error processing problem {problem['task_id']}")
            if problem['is_solved'] == False:
                fail_count += 1
                problems_q.append(problem['task_id'])
            continue
    
    with open('output_data/humaneval/seed/deepseek-coder-v2-lite-instruct/20240828-174550/dscoder_debugged_seeds_deepseek-coder-v2-lite-instruct_1_1_10.jsonl', "r") as f:
        fixed = f.readlines()
    for fix_problem in fixed:
        fix_problem = json.loads(fix_problem)
        if fix_problem['task_id'] in problems_q:
            print(1)

    logger.info(f"Success count: {success_count}")
    logger.info(f"Total problems: {len(problems)}")
    logger.info(f"Unhandled failures: {fail_count}")

if __name__ == "__main__":
    input_seeds = "input_data/humaneval/seed/deepseek-coder-v2-lite-instruct/seed.jsonl"

    with open(input_seeds, "r") as f:
        problems = f.readlines()

    check_all_problems(problems)
    #parse_all_problems(problems)

    # parse the one with 'task_id': 'HumanEval/32'
    # for problem in problems:
    #     problem = json.loads(problem)
    #     if problem['task_id'] == 'HumanEval/33':
    #         parsed_tests = parse_specific_problem(problem)
    #         break
