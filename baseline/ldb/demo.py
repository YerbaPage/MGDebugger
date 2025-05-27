import re
import pandas as pd
import gradio as gr
import ast
import random
import sys
sys.path.append("..")

from programming.generators import PyGenerator, model_factory
from programming.executors import PyExecutor
from programming.utils import *


def read_test_input(test_input):
    pairs = []
    for i, o in zip(test_input["Expression"], test_input["Expected Value"]):
        pairs.append((i, o))
    asserts = []
    for i, o in pairs:
        asserts.append(f"assert {i} == {o}")
    return asserts


def parse_failed_tests(failed_tests):
    pairs = []
    for failed_test in failed_tests:
        failed_test = failed_test.strip()
        pattern = f"assert (.*) == (.*) # Real Execution Output: (.*)"
        match = re.match(pattern, failed_test)
        if match:
            inputs = match.group(1)
            expected = match.group(2)
            actual = match.group(3)
            pairs.append((inputs, expected, actual))
    return pairs


def get_updated_test_df(test_input, failed_tests):
    failed_tests = parse_failed_tests(failed_tests)
    updated_data = []
    for i, o in zip(test_input["Expression"], test_input["Expected Value"]):
        pass_flag = True
        for f_i, f_o, f_a in failed_tests:
            if i == f_i and o == f_o:
                updated_data.append(["Fail", f_i, f_o, f_a])
                pass_flag = False
                break
        if pass_flag:
            updated_data.append(["Pass", i, o, o])
    return pd.DataFrame(
        updated_data, columns=["Pass?", "Expression", "Expected Value", "Actual Value"]
    )


def test_code(code, fixed_code, ori_tests):
    if fixed_code:
        code = fixed_code
        fixed_code = ""

    tests = read_test_input(ori_tests)
    gen = PyGenerator()
    exe = PyExecutor()
    code = IMPORT_HEADER + code
    is_passing, failed_tests, _ = exe.execute(code, tests)
    code = code.replace(IMPORT_HEADER, "").strip()
    fixed_code = fixed_code.replace(IMPORT_HEADER, "").strip()
    updated_test_df = get_updated_test_df(ori_tests, failed_tests)
    return updated_test_df, code, fixed_code


def debug_code(openai_key, model, task, code, fixed_code, ori_tests):
    if fixed_code:
        code = fixed_code
        fixed_code = ""

    tests = read_test_input(ori_tests)
    gen = PyGenerator()
    exe = PyExecutor()
    model = model_factory(model, key=openai_key)
    dataset_type = "HumanEval"

    code = IMPORT_HEADER + code
    is_passing, failed_tests, _ = exe.execute(code, tests)

    if is_passing:
        updated_test_df = get_updated_test_df(ori_tests, failed_tests)
        code = code.replace(IMPORT_HEADER, "").strip()
        return "Program passes all tests", code, code, updated_test_df
    else:
        test = random.sample(failed_tests, 1)[0]
        tree = ast.parse(test)
        entry_point = tree.body[0].test.left.func.id
        func_header = get_func_header(code, entry_point)
        prompt = insert_comment(func_header, task, entry_point)
        code = insert_comment(code, task, entry_point)
        messages = gen.ldb_debug(
            prompt, code, test, entry_point, model, "", dataset_type, "block"
        )
        debug_message = '======== Prompt ========\n'
        for i, m in enumerate(messages):
            if i == 0:
                debug_message += "----- System -----\n" + m.content.strip() + '\n'
            elif i == len(messages) - 1:
                debug_message += '\n======== Response ========\n'
                debug_message += m.content.strip()
            else:
                if i % 2 == 1:
                    debug_message += "----- User -----\n" + m.content.strip() + '\n'
                else:
                    debug_message += "----- Assistant -----\n" + m.content.strip() + '\n'

        fixed_code, messages = gen.ldb_generate(
            func_sig=task,
            model=model,
            prev_func_impl=code,
            messages=messages,
            failed_tests=test,
            dataset_type=dataset_type,
        )
        code = code.replace(IMPORT_HEADER, "").strip()
        fixed_code = fixed_code.replace(IMPORT_HEADER, "").strip()
        is_passing, failed_tests, _ = exe.execute(fixed_code, tests)
        updated_test_df = get_updated_test_df(ori_tests, failed_tests)
        return debug_message, code, fixed_code, updated_test_df


app = gr.Blocks(
    theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue="gray")
)

with app:
    with gr.Row():
        gr.Markdown("# LDB Demo: Debugging with Large Language Model")
        log_checkbox = gr.Checkbox(label="View detailed log", value=False)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                openai_key_input = gr.Textbox(
                    label="OpenAI Key",
                    placeholder="Enter your OpenAI key here",
                    type="password",
                )
                model_selector = gr.Dropdown(
                    label="Choose Model",
                    choices=["gpt-3.5-turbo-0613", "gpt-4-1106-preview"],
                    value="gpt-3.5-turbo-0613",
                )
            task_desc = gr.TextArea(
                label="Task Description",
                placeholder="Enter your task description here",
                lines=3,
            )
            test_input = gr.DataFrame(
                label="Test to Run",
                headers=["Pass?", "Expression", "Expected Value", "Actual Value"],
                interactive=True,
                col_count=(4, "fixed"),
                row_count=(1, "dynamic"),
            )
            with gr.Row():  # This Row will contain the buttons
                test_button = gr.Button("Test", variant="secondary")
                debug_button = gr.Button("Debug", variant="primary")
                clear_button = gr.Button("Clear", variant="neutral")
        with gr.Column():
            code_input = gr.TextArea(
                label="Code Input",
                placeholder="Enter your code here",
                lines=10,
            )
            fixed_code_output = gr.TextArea(
                label="Fixed Code",
                placeholder="Fixed code will be shown here",
                lines=10,
                interactive=False,
                visible=True,
            )

    with gr.Row():
        output_window = gr.TextArea(
            label="Output Window", lines=20, interactive=False, visible=False
        )
    
    def toggle_log_checkbox(is_checked, text):
        if is_checked:
            return gr.update(visible=True, value=text)
        else:
            return gr.update(visible=False, value=text)
    log_checkbox.change(toggle_log_checkbox, [log_checkbox, output_window], output_window)

    test_button.click(
        test_code,
        inputs=[code_input, fixed_code_output, test_input],
        outputs=[test_input, code_input, fixed_code_output],
    )
    debug_button.click(
        debug_code,
        inputs=[
            openai_key_input,
            model_selector,
            task_desc,
            code_input,
            fixed_code_output,
            test_input,
        ],
        outputs=[output_window, code_input, fixed_code_output, test_input],
    )

    def clear_inputs():
        return (
            "",
            "",
            pd.DataFrame(
                {
                    "Pass?": [],
                    "Expression": [],
                    "Expected Value": [],
                    "Actual Value": [],
                }
            ),
            "",
            "",
        )

    clear_button.click(
        clear_inputs,
        inputs=[],
        outputs=[task_desc, code_input, test_input, output_window, fixed_code_output],
    )

    gr.Markdown("## Text Examples")
    gr.Examples(
        [
            [
                "Sum a list",
                pd.DataFrame(
                    {
                        "Pass?": ["?"],
                        "Expression": ["sum_list([1, 2, 3])"],
                        "Expected Value": ["6"],
                        "Actual Value": [""],
                    }
                ),
                "def sum_list(lst):\n    return sum(lst)+1",
            ],
            [
                ("Evaluate whether the given number n can be written as "
                 "the sum of exactly 4 positive even numbers"),
                pd.DataFrame(
                    {
                        "Pass?": ["?", "?", "?"],
                        "Expression": ["is_equal_to_sum_even(4)", "is_equal_to_sum_even(6)", "is_equal_to_sum_even(8)"],
                        "Expected Value": ["False", "False", "True"],
                        "Actual Value": ["", "", ""],
                    }
                ),
                '''\
def is_equal_to_sum_even(n):
    if n % 2 != 0:
        return False
    for i in range(1, n//2 + 1):
        if (n - 2*i) % 2 == 0:
            return True
    return False'''
            ],
            [
                ("Create a function which returns the largest index of an element which "
                 "is not greater than or equal to the element immediately preceding it. If "
                 "no such element exists then return -1. The given array will not contain "
                 "duplicate values."),
                pd.DataFrame(
                    {
                        "Pass?": ["?", "?"],
                        "Expression": ["can_arrange([1,2,4,3,5])", "can_arrange([1,2,3])"],
                        "Expected Value": ["3", "-1"],
                        "Actual Value": ["", ""],
                    }
                ),
                '''\
def can_arrange(arr):
    largest_index = -1
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            largest_index = i-1
    return largest_index'''
            ]
        ],
        inputs=[task_desc, test_input, code_input],
    )


app.launch()
