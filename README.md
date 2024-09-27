# MGDebugger: Multi-Granularity Debugger

## Introduction

Multi Granularity De-bugger (MGDebugger), a hierarchical code debugger by isolating, identifying, and resolving bugs at various levels of granularity.
![1](figures/overview_v1_page.jpg)
![2](figures/subfunction_debug_page.jpg)

## Running MGDebugger

Prerequisites

vllm: Ensure you have vllm installed. Follow the official vllm installation guide (https://github.com/vllm-project/vllm) for detailed instructions.
Running the DeepSeek-Coder-V2-Lite-Instruct VLLM Server

```bash
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code --dtype auto --api-key token-abc123s --port 18889 --max-model-len 40960
```

Once the VLLM server is running
```bash
python main.py
```

After program is finished, you can find all logs in output_data.
## Contribution

We welcome contributions to improve MGDebugger. If you have suggestions, enhancements, or bug reports, please feel free to open an issue or pull request on GitHub.