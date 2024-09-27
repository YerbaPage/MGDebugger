# MG-Coder

## Running 

To run the MG-Coder, you need to have Python 3.8 installed. You can install the required packages by running:

```bash
python chat_mgcoder_framework.py
```

MG-Coder is a debugging system for Python code, utilizing a multistep approach to identify and fix bugs in functions.

## Key Components

1. Hierarchical Code Decomposition
2. Dependency Graph Analysis
3. Subfunction Test Case Generation
4. Bottom-Up Hierarchical Debugging

## Hierarchical Code Decomposition

- Converts original code into a tree-style hierarchical structure
- Breaks down complex functions into smaller, manageable sub-functions

> If the generated codes is nested, we will convert the nested code into a flat structure to make it easier to separate and debug the functions.

Example:
```python
def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    """
    if is_palindrome(string):
        return string
    for i in range(len(string), 0, -1):
        if is_palindrome(string[:i]):
            return string + string[i-1::-1]
    return string + string[::-1]
```

## Hierarchical Code Decomposition

The previous code will be converted to:

```python
def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]

def find_longest_palindromic_postfix(string: str) -> str:
    """ Find the longest postfix of the supplied string that is a palindrome. """
    for i in range(len(string), 0, -1):
        if is_palindrome(string[:i]):
            return string[:i]
    return ""

def reverse_prefix(string: str, prefix_length: int) -> str:
    """ Reverse the prefix of the string up to the given length. """
    return string[:prefix_length][::-1]

def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    """
    if is_palindrome(string):
        return string
    
    longest_postfix = find_longest_palindromic_postfix(string)
    prefix_length = len(string) - len(longest_postfix)
    prefix_to_reverse = string[:prefix_length]
    reversed_prefix = prefix_to_reverse[::-1]
    
    return string + reversed_prefix
```

## Dependency Graph Analysis

We will analyze the code structure and dependencies to sort the functions from bottom to top for debugging.

```
├── reverse_prefix
└── make_palindrome
    ├── find_longest_palindromic_postfix
    │   └── is_palindrome
    └── is_palindrome
```

> As you can see, the `reverse_prefix` function is not used by make_palindrome, so we may need to remove it. (Since this may not be a common case,we haven't put effort into this part yet.)

## Test Case Generation

- Generates specific test cases for each function
- Ensures consistency with provided gold test cases (We use a majority voting system to ensure the correctness of the test cases)

Example for `find_longest_palindromic_postfix`:
```json
{
    "test_cases": [
        {"input": {"string": ""}, "expected_output": ""},
        {"input": {"string": "cat"}, "expected_output": "t"},
        {"input": {"string": "cata"}, "expected_output": "ata"}
    ]
}
```

> In our experiments, we found that generating test cases for functions is challenging. We tried simple majority voting but the results are not satisfactory. We will continue to consider improving this part.

## Iterative Debugging

- Debugs each function individually, starting from the bottom to top
- Ask the LLM to analyzes code execution step-by-step provided with results from test cases
- Identifies bugs and proposes fixes

## Code Reconstruction

- Merges fixed sub-functions back into the main code from bottom to top
- Reconstructs the full, debugged code

## Final Fixed Code

```python
def is_palindrome(string: str) -> bool:
    """Test if given string is a palindrome """
    return string == string[::-1]

def find_longest_palindromic_postfix(string: str) -> str:
    """Find the longest postfix of the supplied string that is a palindrome. """
    for i in range(len(string), 0, -1):
        if is_palindrome(string[:i]):
            return string[:i]
    return ''

def reverse_prefix(string: str, prefix_length: int) -> str:
    """Reverse the prefix of the string up to the given length. """
    if prefix_length > len(string):
        prefix_length = len(string)
    return string[:prefix_length][::-1] + string[prefix_length:]

def make_palindrome(string: str) -> str:
    """Find the shortest palindrome that begins with a supplied string.
Algorithm idea is simple:
- Find the longest postfix of supplied string that is a palindrome.
- Append to the end of the string reverse of a string prefix that comes before the palindromic suffix."""
    if is_palindrome(string):
        return string
    longest_postfix = find_longest_palindromic_postfix(string)
    prefix_length = len(string) - len(longest_postfix)
    prefix_to_reverse = string[:prefix_length]
    reversed_prefix = prefix_to_reverse[::-1]
    return string + reversed_prefix
```
