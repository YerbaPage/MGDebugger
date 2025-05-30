import time
from openai import OpenAI
from loguru import logger
from config import MODEL, LLM_BASE_URL, LLM_API_KEY, MAX_VLLM_RETRIES, RETRY_DELAY, TEMPERATURE, MINP
import config

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


def get_completion_with_retry(messages, model=MODEL, max_retries=MAX_VLLM_RETRIES):
    """Get completion from LLM with retry mechanism"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting LLM call (attempt {attempt + 1}/{max_retries})")
            logger.info(f"Input messages: {messages[-1]['content']}")
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=TEMPERATURE,
                extra_body={"min_p": MINP}
            )
            response = chat_completion.choices[0].message.content
            logger.info(f"LLM response: {response}")
            logger.info("LLM call received")

            # Update token counts
            config.TOTAL_PROMPT_TOKENS += chat_completion.usage.prompt_tokens
            config.TOTAL_COMPLETION_TOKENS += chat_completion.usage.completion_tokens

            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries reached. Giving up.")
                raise 