# Configuration and hyperparameters for MGDebugger

MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

# LLM Client Configuration
LLM_BASE_URL = "http://localhost:18889/v1"
LLM_API_KEY = "token-abc123s"

# Hyperparameters
MAX_VLLM_RETRIES = 10  # maximum number of retries for the VLLM call
MAX_PARSE_RETRIES = 3  # we sometimes fail to parse the code/test cases from the response, so we retry
MAX_DEBUG_RETRIES = 3  # 3 seems to be slightly better than 1, but the difference is not significant
RETRY_DELAY = 0  # seconds
REPEAT_CONVERT_HIERARCHICAL_NUM = 1  # seems unimportant
REPEAT_TEST_CASE_GENERATION_NUM = 1  # 1 seems to be better than 3
MAX_OUTER_RETRY = 10  # maximum number of retries for the entire debugging process
CONTINUOUS_RETRY = False  # whether to retry from the last fixed code, else retry from the original buggy code. it seems better to set this to False
TEMPERATURE = 0.8  # 0.8 better than 1.0 better than 0.2
MINP = 0.05

# Global statistics tracking
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_MG_DEBUG_CALLS = 0
TOTAL_DEBUG_FUNCTION_CALLS = 0
TOTAL_GENERATE_TEST_CASES_CALLS = 0
TOTAL_CONVERT_HIERARCHICAL_CALLS = 0 