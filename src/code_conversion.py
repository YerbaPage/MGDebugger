from loguru import logger
from llm_client import get_completion_with_retry
from utils import split_nested_functions, extract_code_blocks, extract_functions
from config import MODEL, REPEAT_CONVERT_HIERARCHICAL_NUM
import config


def convert_to_hierarchical(code, include_example=False):
    """Convert code to hierarchical structure with multiple levels of sub-functions"""
    config.TOTAL_CONVERT_HIERARCHICAL_CALLS += 1

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