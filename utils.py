import ast
from loguru import logger
import sys
import io
import json
import re
import traceback
import os
from timeout_utils import function_with_timeout

helpers = [
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import statistics",
    "import functools",
    "import hashlib",
    "import numpy",
    "import numpy as np",
    "import string",
    "from typing import *",
    "from collections import *",
    "import heapq as hq",
    "from itertools import *",
    "from math import *",
    "from statistics import *",
    "from functools import *",
    "from collections import *",
    "from datetime import *",
    "from copy import *",
]

STARTING_CODE = "\n".join(helpers)


def create_dependency_graph(functions):
    graph = {func_name: set() for func_name in functions}
    for func_name, func_code in functions.items():
        for other_func in functions:
            if other_func in func_code and other_func != func_name:
                graph[func_name].add(other_func)
    return graph


def topological_sort(graph):
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return stack


def merge_changes_to_parents(func_name, dependency_graph, functions):
    # Update the function in the functions dictionary
    logger.info(f"Updating function {func_name} in the functions dictionary")

    # For any function that calls the modified function, update its code
    for parent, children in dependency_graph.items():
        if func_name in children:
            parent_code = functions[parent]
            updated_parent_code = parent_code.replace(func_name, f"{func_name}")
            functions[parent] = updated_parent_code
            logger.info(f"Updated references to {func_name} in parent function {parent}")

    # Regenerate the full code
    full_code = "\n\n".join(functions.values())

    logger.info(f"Merged changes from {func_name} to all relevant functions")
    return full_code


def extract_functions(code):
    logger.info("Extracting functions from code")
    tree = ast.parse(code)
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(code, node)
            functions[node.name] = func_code
    logger.info(f"Extracted {len(functions)} functions: {', '.join(functions.keys())}")
    return functions


def extract_code_blocks(response):
    """Extract all code blocks from the response."""
    return re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)


def extract_function(code_block, function_name):
    """Extract a specific function from a code block."""
    try:
        tree = ast.parse(code_block)
    except:
        logger.error(f"Failed to parse code block for function: {function_name} from\n{code_block}")
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_source_segment(code_block, node)
    return None


def evaluate_given_tests(code, given_tests, max_memory=100 * 1024 * 1024):
    test_code = f"{STARTING_CODE}\n\n{code}\n\n{given_tests}"
    try:
        function_with_timeout(exec, (test_code, globals()), timeout=10, max_memory=max_memory)
        return True
    except TimeoutError as e:
        logger.error(f"Timeout Error: {str(e)}")
    except MemoryError as e:
        logger.error(f"Memory Error: {str(e)}")
    except AssertionError as e:
        logger.error(f"Assertion Error: {str(e)}")
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        logger.error(f'Traceback: {traceback.format_exc()}')
    return False

def evaluate_simple(code, entry_point, all_test, max_memory=100 * 1024 * 1024):
    '''
    directly concatenate the code and test code to evaluate on the private test cases
    '''

    test_code = f"{STARTING_CODE}\n\n{code}\n\n{all_test}\n\ncheck({entry_point})"
    try:
        function_with_timeout(exec, (test_code, globals()), timeout=10, max_memory=max_memory)
        return True
    except TimeoutError as e:
        logger.error(f"Timeout Error: {str(e)}")
    except MemoryError as e:
        logger.error(f"Memory Error: {str(e)}")
    except AssertionError as e:
        logger.error(f"Assertion Error: {str(e)}")
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        logger.error(f'Traceback: {traceback.format_exc()}')
    return False


def evaluate(code, entry_point, testcase, return_trace=False):
    logger.info(f"Evaluating {entry_point} with testcase: {testcase['input']}")

    # Extract all functions from the code
    try:
        functions = extract_functions(code)
    except:
        logger.error(f"Failed to extract functions from code {code}")
        # import pdb
        # pdb.set_trace()
    logger.info(f"Extracted functions: {', '.join(functions.keys())}")

    # filter the functions that are called in the entry_point function
    entry_point_function = functions[entry_point]
    # entry_point_tree = ast.parse(entry_point_function)
    # entry_point_calls = [node.func.id for node in ast.walk(entry_point_tree) if isinstance(node, ast.Call)]
    # functions = {name: func for name, func in functions.items() if name in entry_point_calls}
    # directly search for the string
    functions = {name: func for name, func in functions.items() if name in entry_point_function}
    logger.info(f"Filtered functions: {', '.join(functions.keys())}")

    # Combine all functions into a single code block
    full_code = "\n\n".join(functions.values())
    # logger.info(f"Code being evaluated:\n{full_code}")

    # Convert the input to a string representation that can be safely evaluated
    input_repr = repr(testcase['input'])

    if isinstance(testcase['input'], dict):
        # Sometimes the input is a dictionary, which needs to be unpacked as keyword arguments
        test_code = f'''{full_code}\n\nprint(repr({entry_point}(**{input_repr})))'''
    else:
        test_code = f'''{full_code}\n\nprint(repr({entry_point}({input_repr})))'''

    # add the starting code to the test code
    test_code = f"{STARTING_CODE}\n\n{test_code}"

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        function_with_timeout(exec, (test_code, globals()), timeout=10)
        output = new_stdout.getvalue().strip()
        sys.stdout = old_stdout

        # Convert both expected and actual output to the same type for comparison
        expected_output = repr(testcase["expected_output"])

        # Update actual_output before assertion
        testcase['actual_output'] = ast.literal_eval(output)

        assert output == expected_output, f"Expected {expected_output}, but got {output}"
        logger.info(f'Test case passed: {testcase}')
        logger.info(f'Expected: {expected_output}, Got: {output}')
        return True, testcase
    except TimeoutError:
        logger.error(f'Test case failed: {testcase}')
        logger.error(f"Timeout Error: {str(e)}")
    except AssertionError as e:
        logger.error(f'Test case failed: {testcase}')
        logger.error(str(e))
    except Exception as e:
        logger.error(f'Test case failed: {testcase}')
        logger.error(f'Error: {str(e)}')
        logger.error(f'Traceback: {traceback.format_exc()}')
        testcase['actual_output'] = str(e)
        if return_trace:
            testcase['traceback'] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return False, testcase


def extract_json_from_string(s):

    # search for all the ```json blocks
    matches = re.findall(r'```json\s*(.*?)\s*```', s, re.DOTALL)
    if matches:
        return matches[-1]
    return None


def parse_json_response(response):
    json_str = extract_json_from_string(response)
    if json_str:
        try:
            # Standard JSON corrections
            json_str = json_str.strip().replace("True", "true")
            json_str = json_str.replace("False", "false")
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace("None", "null")

            # Convert tuple notation to list notation
            json_str = re.sub(r'\((-?\d+),\s*(-?\d+)\)', r'[\1, \2]', json_str)

            logger.info(f"Extracted JSON string: {json_str}")
            try:
                return json.loads(json_str)
            except:
                # remove comments (for mistral model)
                json_str = re.sub(r'#.*', '', json_str)
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON: {json_str}")
            logger.error(f"JSONDecodeError: {str(e)}")
            # import pdb
            # pdb.set_trace()
    else:
        logger.error("No JSON object found in the response")
    return None


def get_dependency_graph_str(graph, root=None, prefix="", is_last=True):
    result = []

    if root is None:
        # Collect all roots if no specific root is given
        roots = [node for node in graph if not any(node in children for children in graph.values())]
        for i, root in enumerate(roots):
            result.append(get_dependency_graph_str(graph, root, "", i == len(roots) - 1))
        return "\n".join(result)

    connector = "└── " if is_last else "├── "
    result.append(prefix + connector + root)

    if root in graph:
        children = sorted(graph[root])
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            result.append(get_dependency_graph_str(graph, child, new_prefix, is_last_child))

    return "\n".join(result)


def extract_functions_from_code(node, parent=None):
    """ Recursively extract functions and set parents. """
    if isinstance(node, ast.Module):
        for n in node.body:
            extract_functions_from_code(n, parent=node)
    elif isinstance(node, ast.FunctionDef):
        node.parent = parent
        if parent is not None and isinstance(parent, (ast.FunctionDef, ast.Module)):
            parent.children.append(node)
        for n in node.body:
            extract_functions_from_code(n, parent=node)


def split_nested_functions(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        node.children = []
    extract_functions_from_code(tree)

    flat_functions = []

    def flatten_functions(node):
        if isinstance(node, ast.FunctionDef):
            flat_functions.append(node)
            # Remove nested function definitions from the body
            node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef)]
        for child in node.children:
            flatten_functions(child)

    flatten_functions(tree)

    # Function to correct indentation for function docstrings
    def correct_indentation(functions):
        for func in functions:
            # Get existing docstring if present
            docstring = ast.get_docstring(func)
            if docstring:
                # Replace existing docstring node with corrected indentation
                corrected_docstring = "\n".join([line if line.strip() != "" else "" for line in docstring.split("\n")])
                func.body[0].value.s = corrected_docstring

    correct_indentation(flat_functions)

    return '\n\n'.join(ast.unparse(f).strip() for f in flat_functions)


def remove_unused_functions(code, entry_point):

    tree = ast.parse(code)

    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    function_calls = set()

    class FunctionCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in function_names:
                function_calls.add(node.func.id)
            self.generic_visit(node)

    FunctionCallVisitor().visit(tree)

    used_functions = set()

    def mark_used(func_name):
        if func_name not in used_functions:
            used_functions.add(func_name)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    FunctionCallVisitor().visit(node)
                    for call in function_calls:
                        mark_used(call)

    mark_used(entry_point)

    # only keep the functions that are used
    tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name in used_functions]

    all_unused_functions = function_names - used_functions

    # convert back to code
    return ast.unparse(tree), all_unused_functions


def test_remove_unused_functions():
    code = '''
def rolling_max(numbers: List[int]) -> List[int]:
    """From a given list of integers, generate a list of rolling maximum element found until given moment
in the sequence.
>>> rolling_max([1, 2, 3, 2, 3, 4, 2])
[1, 2, 3, 3, 3, 4, 4]"""
    (max_so_far, rolling_max_list) = initialize_max_and_list(numbers)
    for num in numbers[1:]:
        (max_so_far, rolling_max_list) = update_max_and_list(max_so_far, num, rolling_max_list)
    return rolling_max_list

def initialize_max_and_list(numbers: List[int]) -> Tuple[int, List[int]]:
    max_so_far = numbers[0]
    rolling_max_list = [max_so_far]
    return (max_so_far, rolling_max_list)

def update_max_and_list(max_so_far: int, num: int, rolling_max_list: List[int]) -> Tuple[int, List[int]]:
    max_so_far = max(max_so_far, num)
    rolling_max_list.append(max_so_far)
    return (max_so_far, rolling_max_list)
    
def clean_data(data: List[str]) -> List[str]:
    return [d.strip() for d in data]
    '''.strip()

    entry_point = "rolling_max"
    logger.info(f"Original code:\n{code}")
    output, unused_functions = remove_unused_functions(code, entry_point)
    logger.info(f"Unused functions: {unused_functions}")
    logger.info(f"Cleaned code:\n{output}")


def test_split_nested_functions():
    # The initial code provided by the user
    code = '''
def find_suffix_start(s: str) -> int:
    for i in range(len(s)):
        if is_palindrome(s[i:]):
            return i
    return 0

def make_palindrome(string: str) -> str:
    """This function takes a string and returns a palindrome by appending the reverse of the prefix of the string that makes it a palindrome."""
    
    
    def is_palindrome(s: str) -> bool:
        """
        This function takes a string and returns True if it is a palindrome, False otherwise.
        """
    
    
        def compare(s: str) -> bool:
            """
            This function takes a string and returns True if it is a palindrome, False otherwise.
            inner function
            """

            return s == s[::-1]
    
        return compare(s)

    suffix_start = find_suffix_start(string)
    return string + string[:suffix_start][::-1]
    '''.strip()

    # Splitting the nested functions and correcting the indentation
    output = split_nested_functions(code)
    print(output)


def test_parse_json_response():

    response = """
**All Test Cases:**
```json
{
    "test_cases": [
        {"input": {"date": "03-11-2000"}, "expected_output": [11, 3, 2000]},
        {"input": {"date": "15-01-2012"}, "expected_output": [15, 1, 2012]},
        {"input": {"date": "04-0-2040"}, "expected_output": None},
        {"input": {"date": "06-04-2020"}, "expected_output": [4, 6, 2020]},
        {"input": {"date": "06/04/2020"}, "expected_output": None}
    ]
}
```
    """.strip()

    parsed_json = parse_json_response(response)
    print(parsed_json)


def insert_docstring(code, docstring):

    # surround the docstring with triple quotes
    docstring = f'"""{docstring}"""'

    lines = code.split('\n')
    # Find the first non-empty line
    first_line = next((i for i, line in enumerate(lines) if line.strip()), 0)

    # Determine the indentation of the first line
    indentation = len(lines[first_line]) - len(lines[first_line].lstrip())

    # Find the 'def' line
    def_line = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), first_line)

    # Insert the docstring after the 'def' line, maintaining indentation
    docstring_lines = [' ' * (indentation + 4) + line for line in docstring.split('\n')]
    lines = lines[:def_line+1] + docstring_lines + lines[def_line+1:]

    return '\n'.join(lines)


def parse_transcoder_problem_content(problem):
    # Extract the last group of content between [c++] and [python]
    cpp_code = problem["prompt"].split("[c++]")[-1].split("[python]")[0].strip()
    full_question = f'This function is translated into Python from the following C++ code: \n{cpp_code}\n'

    try:
        # Try to parse the existing solution
        tree = ast.parse(problem["solution"])

        # Create a new docstring node
        docstring = ast.Expr(ast.Str(full_question))

        # Find the first function definition in the AST
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Insert the docstring at the beginning of the function body
                node.body.insert(0, docstring)
                break
        else:
            # If no function definition is found, add the docstring at the end of the module
            tree.body.append(docstring)

        # Convert the modified AST back to source code
        modified_solution = ast.unparse(tree)

    except SyntaxError:
        # If there's a syntax error, use the string-based method
        logger.debug(f"Failed to parse solution for problem: {problem['task_id']}")
        modified_solution = insert_docstring(problem["solution"], full_question)
        logger.debug(f"Modified solution: {modified_solution}")

    # Update the problem dictionary with the modified solution
    problem["solution"] = modified_solution

    return problem


def test_parse_transcoder_problem_content():

    input_seeds = "input_data/transcoder/seed/starcoder/seed.jsonl"
    with open(input_seeds, "r") as f:
        problems = [json.loads(line) for line in f]

    for problem in problems:
        try:
            result = parse_transcoder_problem_content(problem)
        except Exception as e:
            logger.error(f"Failed to parse solution for problem: {problem['task_id']}")
            logger.error(f"The solution is: \n{problem['solution']}")
            logger.error(f"Error: {str(e)}")

    logger.info("Successfully parsed all solutions")
    # show an example
    logger.info(f"Example result: {result['solution']}")


if __name__ == "__main__":

    # test_split_nested_functions()
    # test_parse_json_response()
    # test_remove_unused_functions()
    test_parse_transcoder_problem_content()
