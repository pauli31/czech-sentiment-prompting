# -*- coding: utf-8 -*-

import logging
import os

from hugchat import hugchat
from hugchat.login import Login

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, DEFAULT_CREDENTIALS_FILE_PATH_LLAMA

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

TEST_REVIEW = "\"Jedna z vyjímek, kdy filmové zpracování je lepší než knižní předloha. Tu (od Winstona Grooma) jsem četl až po shlédnutí filmu, který jsem viděl poprve v roce 1994, snad týden po Pulp Fiction ( tak silné nabídky v kinech se myslím jen tak nedočkáme), a příběh v ní tak pěkně neplyne, jsou tam i věci jako, že Forest letí na měsíc či je přeborníkem v zápase řeckořímském, prostě film po stránce příběhové je určitě lepší. Jako celek (režie, scénář, herecké výkony, triky) pak patří mezi to nejlepší co se na konci 20 století natočilo a i při vícenásobném zhlédnutí nijak nenudí. Zase kdyby knihy nebylo, tak by nebylo ani Foresta Gumpa...\""


def main():
    print("Hello")

    username, pwd = load_credentials()
    chatbot = init_hugging_chat(username, pwd)
    chatbot.switch_llm(1)  # Switch to `meta-llama/Llama-2-70b-chat-hf`

    classify_sentiment(chatbot, "You are a sentiment classifier, classify the following review as \"positive\", "
                                "\"negative\" or \"neutral\". Answer only one word.\n\n The review:\n\n{text}",
                       TEST_REVIEW)

    for i in range(10):
        # time.sleep(1)
        classify_sentiment(chatbot, "You are a sentiment classifier, classify the following review as \"positive\", "
                                    "\"negative\" or \"neutral\". Answer only one word.\n\n The review:\n\n{text}",
                           TEST_REVIEW)

    conv_list = chatbot.get_conversation_list()
    for conv in conv_list:
        logger.info("Deleting conversation:" + str(conv))
        chatbot.delete_conversation(conv)


def classify_sentiment(chatbot: hugchat.ChatBot, prompt: str, message: str, create_new_conversation: bool = True,
                       delete_conversation: bool = True):
    # Create a new conversation
    id_conv = None
    if create_new_conversation:
        id_conv = chatbot.new_conversation()
        chatbot.change_conversation(id_conv)

    text = prompt.format(text=message)
    result = chatbot.chat(text)
    # logger.info("Result:" + result)

    if delete_conversation and id_conv is not None:
        chatbot.delete_conversation(id_conv)

    # result = "positive"

    return result


def load_credentials(credentials_path: str = DEFAULT_CREDENTIALS_FILE_PATH_LLAMA) -> [str, str]:
    with open(credentials_path) as f:
        username = f.readline().strip()
        pwd = f.readline().strip()

    return username, pwd


def init_hugging_chat(username, pwd):
    # Log in to huggingface and grant authorization to huggingchat
    sign = Login(username, pwd)

    # Save cookies to the local directory
    cookie_path_dir = "./cookies_snapshot"
    if os.path.exists(cookie_path_dir):
        # Load cookies when you restart your program:
        cookies = sign.loadCookiesFromDir(
            cookie_path_dir)
        # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.
    else:
        cookies = sign.login()
        sign.saveCookiesToDir(cookie_path_dir)

    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
    chatbot.switch_llm(1)  # Switch to `meta-llama/Llama-2-70b-chat-hf`
    logger.info("Credentials loaded and chatbot created")

    # chatbot = None
    return chatbot


if __name__ == '__main__':
    main()
