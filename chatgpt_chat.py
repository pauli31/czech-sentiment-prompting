import logging
from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT
import openai

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)


def classify_sentiment_chatgpt(model_type,
                               prompt: str,
                               user_prompt: str,
                               message: str,
                               create_new_conversation: bool = True,
                               delete_conversation: bool = True,
                               temperature: float = 0.9,
                               max_tokens: int = 1024,
                               top_p: float = 0.95):
    text = user_prompt.format(text=message)
    msgs = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': text},
    ]

    response = openai.ChatCompletion.create(
        model=model_type,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    result = response.choices[0].message["content"]
    return result


def load_credentials_open_ai(credentials_path: str = DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT):
    with open(credentials_path) as f:
        key = f.readline().strip()

    return key


def init_open_ai(api_key:str):
    openai.api_key = api_key
    logger.info("Api key set")
