# MGDebugger: Multi-Granularity Debugger

## Introduction

MGDebugger is a hierarchical code debugger designed to isolate, identify, and resolve bugs at various levels of granularity. By employing a hierarchical bottom-up debugging approach, it allows for more precise and effective error correction.

![Overview](figures/overview_v1_page.jpg)

![Subfunction Debugging](figures/subfunction_debug_page.jpg)

## Getting Started with MGDebugger

### Prerequisites

Before running MGDebugger, ensure that you have the following prerequisites:

- **vllm**: Make sure vllm is installed on your system. You can follow the official [vllm installation guide](https://github.com/vllm-project/vllm) for detailed instructions.

### Running the DeepSeek-Coder-V2-Lite-Instruct VLLM Server

To start the VLLM server, use the following command:

```bash
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code --dtype auto --api-key token-abc123s --port 18889 --max-model-len 40960
```

### Launching MGDebugger

Once the VLLM server is running, you can start MGDebugger with:

```bash
python main.py
```

> You can adjust the `MODEL` and `input_seeds` in the code if you want to test different models and inputs.

### Logs

After the program finishes running, all logs will be available in the `output_data` directory.

## Contribution

We welcome contributions to enhance MGDebugger! If you have suggestions, improvements, or bug reports, please feel free to open an issue or submit a pull request on GitHub.