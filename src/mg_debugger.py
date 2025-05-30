from loguru import logger
from utils import (
    extract_functions, get_dependency_graph_str, evaluate, 
    create_dependency_graph, topological_sort, merge_changes_to_parents
)
from code_conversion import convert_to_hierarchical
from test_generation import generate_test_cases
from function_debugger import debug_function
from config import MAX_DEBUG_RETRIES, MAX_PARSE_RETRIES
import config


def mg_debug(full_code, gold_test_cases, max_debug_attempts=MAX_DEBUG_RETRIES):
    """Main debugging function using hierarchical approach"""
    config.TOTAL_MG_DEBUG_CALLS += 1
    logger.info("Starting main debugging process")

    convert_hierarchical_attempts = 0
    while convert_hierarchical_attempts < MAX_PARSE_RETRIES:
        try:
            # Convert to tree-style hierarchical structure
            hierarchical_code = convert_to_hierarchical(full_code, include_example=False)
            logger.info(f"Converted code to tree-style hierarchical structure:\n{hierarchical_code}")

            functions = extract_functions(hierarchical_code)

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