# Baseline Methods for Code Debugging

This directory contains implementations of five baseline methods for automated code debugging, separated into individual files for better organization and maintainability.

## Overview

The baseline methods are compared against MGDebugger and include both simple approaches and more sophisticated debugging techniques from recent literature. Each baseline is implemented in a separate Python file with clear documentation.

## Baseline Methods

### 1. No-Debugging Baseline (`no_debugging.py`)
- **Function**: `repeat_generation()`
- **Description**: Generates seed buggy solutions for other debugging baselines to repair. This baseline simply regenerates code from the original problem prompt without any debugging feedback.
- **Implementation**: We follow the codes from LDB~\cite{zhong2024debug} to set up this baseline.
- **Use Case**: Provides a lower bound for comparison and generates initial buggy solutions.

### 2. Simple Feedback Baseline (`simple_feedback.py`)
- **Function**: `simple_baseline()`
- **Description**: Collects minimal feedback by providing only the buggy code and basic error information to the model.
- **Implementation**: Prompts the LLM with just the buggy code and asks for a fix without additional context.
- **Use Case**: Tests the model's ability to debug with minimal information.

### 3. Self-Edit Baseline (`self_edit.py`)
- **Function**: `ut_baseline()`
- **Description**: Collects feedback through unit test execution results and expected vs. actual outputs for failing test cases.
- **Implementation**: We implement this baseline following the methodology described in Zhang et al.~\cite{zhang2023selfedit}, where we execute the generated code on example test cases and embed the results as comments.
- **Use Case**: Leverages unit test feedback for debugging guidance.

### 4. Self-Debugging (Explanation) Baseline (`self_debugging_explanation.py`)
- **Function**: `ut_expl_baseline()`
- **Description**: Collects comprehensive feedback including complete unit test execution results, detailed execution traces, and error messages.
- **Implementation**: We implement this baseline following the approach described in Chen et al.~\cite{chen2023teaching}, implementing the generation, explanation, and feedback steps, using unit tests to produce feedback and leveraging LLMs for explanation.
- **Use Case**: Tests step-by-step explanation-based debugging approach.

### 5. Self-Debugging (Trace) Baseline (`self_debugging_trace.py`)
- **Function**: `ut_trace_baseline()`
- **Description**: Collects feedback through step-by-step execution traces showing intermediate variable values and program execution flow.
- **Implementation**: This variant follows the trace-based debugging approach described in Chen et al.~\cite{chen2023teaching}.
- **Use Case**: Tests execution trace-based debugging approach.

## Common Utilities (`common.py`)

Contains shared functions and configurations used by all baseline methods:
- **LLM Communication**: `get_completion_with_retry()` - Handles API calls with retry logic
- **Execution Tracing**: `collect_execution_traces()` - Collects test execution results
- **Code Extraction**: `extract_fixed_code()` - Extracts code blocks from LLM responses
- **Configuration**: Model settings, retry parameters, and global statistics

## Usage

### Running Individual Baselines

Each baseline can be imported and used independently:

```python
from no_debugging import repeat_generation
from simple_feedback import simple_baseline
from self_edit import ut_baseline
from self_debugging_explanation import ut_expl_baseline
from self_debugging_trace import ut_trace_baseline

# Example usage
fixed_code = simple_baseline(buggy_code, entry_point)
```

### Running with the Baseline Runner

Use `baseline_runner.py` for a unified interface to run any baseline:

```python
from baseline_runner import debug_with_baseline

# Run a specific baseline
fixed_problems, total_unsolved = debug_with_baseline(
    input_seeds="path/to/seeds.jsonl",
    baseline_type="simple",  # Options: "repeat", "simple", "ut", "ut_expl", "ut_trace"
    max_examples=100,
    output_folder="./results"
)
```

### Command Line Usage

```bash
# Edit the configuration in baseline_runner.py
python baseline_runner.py
```

## Configuration

Key parameters in `common.py`:
- `MODEL`: LLM model to use (default: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
- `MAX_OUTER_RETRY`: Maximum retry attempts (default: 10)
- `CONTINUOUS_RETRY`: Whether to use previous attempt's output as input for next retry
- `TEMPERATURE`: LLM temperature setting (default: 0)

## File Structure

```
baseline/
├── README.md                           # This file
├── common.py                          # Shared utilities and configuration
├── no_debugging.py                    # No-debugging baseline
├── simple_feedback.py                 # Simple feedback baseline
├── self_edit.py                       # Self-Edit baseline
├── self_debugging_explanation.py      # Self-Debugging (Explanation) baseline
├── self_debugging_trace.py           # Self-Debugging (Trace) baseline
├── baseline_runner.py                 # Unified runner (to be created)
├── baselines.py                       # Original monolithic file (deprecated)
├── ldb/                              # LDB baseline implementation
└── reflexion/                        # Reflexion baseline implementation
```

## Dependencies

- `loguru`: For logging
- `openai`: For LLM API calls
- `tqdm`: For progress bars
- `typing`: For type hints

## Output

Each baseline generates:
- **Results file**: `{baseline_type}_baseline_unsolved.jsonl`
- **Statistics**: `{baseline_type}_statistics.json`
- **Logs**: `all_info.log`
- **Parameters**: `params.json`

## Citation References

- **LDB**: Zhong et al. (2024) - `zhong2024debug`
- **Reflexion**: Shinn et al. (2023) - `shinn2023reflexion`
- **Self-Edit**: Zhang et al. (2023) - `zhang2023selfedit`
- **Self-Debugging**: Chen et al. (2023) - `chen2023teaching`

## Notes

- For LDB and Reflexion, we use their official codebases
- For Self-Edit and Self-Debugging, we closely follow the implementation details and procedures described in their respective papers to replicate the methods
- Our implementations are available under the baseline folder in the replication package
- The original monolithic `baselines.py` file is kept for reference but is deprecated in favor of the separated implementation 