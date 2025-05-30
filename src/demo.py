from loguru import logger
from main import mg_debug
from openai import OpenAI

MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# MODEL = "TechxGenus/Codestral-22B-v0.1-GPTQ"
# MODEL = "Qwen/CodeQwen1.5-7B-Chat"


client = OpenAI(
    base_url="http://localhost:18889/v1",
    api_key="token-abc123s",
)


def run_demo():
    # Example buggy code
    buggy_code = '''
def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    note_map = {'o': 3, 'o|': 2, '.|': 1}
    return [note_map[x] for x in music_string.split(' ') if x]
    '''.strip()

    # Test cases
    gold_test_cases = [
        {'input': {'music_string': 'o o| .| o| o| .| .| .| .| o o'}, 'expected_output': [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]}
    ]

    logger.info("=== Starting Demo ===")
    logger.info("Original buggy code:")
    logger.info(buggy_code)
    logger.info("\nTest cases:")
    for i, test in enumerate(gold_test_cases, 1):
        logger.info(f"Test {i}: Input = {test['input']}, Expected Output = {test['expected_output']}")

    logger.info("\nAttempting to debug the code...")
    fixed_code = mg_debug(buggy_code, gold_test_cases)

    logger.info("\n=== Results ===")
    logger.info("Fixed code:")
    logger.info(fixed_code)


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add("demo_output.log", rotation="500 MB")
    logger.add(lambda msg: print(msg), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    run_demo()
