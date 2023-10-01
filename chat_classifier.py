import logging

from sklearn.metrics import classification_report

from chatgpt_chat import init_open_ai, load_credentials_open_ai, classify_sentiment_chatgpt
from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, DEFAULT_CREDENTIALS_FILE_PATH_LLAMA, \
    DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT, RANDOM_SEED
from dataset import build_data_loader
from llama_chat import load_credentials, init_hugging_chat, classify_sentiment
from loader import DATASET_LOADERS, DatasetLoader
from utils import print_labels_distribution, print_time_info, evaluate_predictions, get_table_result_string
from abc import ABC, abstractmethod
from tqdm import tqdm
import csv
import time
import wandb
import pandas as pd
from prompt_templates import BASIC_PROMPT, BASIC_BINARY_PROMPT, ADVANCED_PROMPT, ADVANCED_PROMPT_BINARY, \
    PROMPT_DELIMITER, IN_CONTEXT_EXAMPLE_TEMPLATE, IN_CONTEXT_LEARNING_PROMPT, IN_CONTEXT_LEARNING_PROMPT_BINARY

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

FAST_DEBUG_PREDICTIONS = False
NUM_FAST_DEBUG_ITER = 10


class ChatClassifier(ABC):
    def __init__(self, args):
        self.args = args
        self.dataset_loader = DATASET_LOADERS[args.dataset_name](args.max_train_data, args.max_test_data, args.binary)
        self.batch_size = args.batch_size
        self.use_only_train_data = args.use_only_train_data
        self.disable_creating_new_conversation = args.disable_creating_new_conversation
        self.sleep_time_s = args.sleep_time_s
        self.max_retry_attempts = 10
        self.sleep_time_retry_attempt_s = 30

        # Load dataset
        logger.info("Loading dataset")
        if self.use_only_train_data:
            self.train_size = len(self.dataset_loader.get_train_dev_data())
            self.dev_size = 0
        else:
            self.train_size = len(self.dataset_loader.get_train_data())
            self.dev_size = len(self.dataset_loader.get_dev_data())

        self.test_size = len(self.dataset_loader.get_test_data())
        self.num_labels = self.dataset_loader.get_class_num()

        if self.use_only_train_data:
            self.train_data_loader: DatasetLoader = build_data_loader(self.dataset_loader.get_train_dev_data(),
                                                                      self.batch_size)
            self.dev_data_loader = None
        else:
            self.train_data_loader: DatasetLoader = build_data_loader(self.dataset_loader.get_train_data(),
                                                                      self.batch_size)
            self.dev_data_loader = build_data_loader(self.dataset_loader.get_dev_data(), self.batch_size)

        self.test_data_loader = build_data_loader(self.dataset_loader.get_test_data(), self.batch_size)

        if args.dataset_name == 'imdb-csfd':
            self.dev_data_czech_loader = build_data_loader(self.dataset_loader.get_dev_data_czech(), self.batch_size)
            self.dev_czech_size = len(self.dataset_loader.get_dev_data_czech())

        if args.dataset_name == 'csfd-imdb':
            self.dev_data_eng_loader = build_data_loader(self.dataset_loader.get_dev_data_eng(), self.batch_size)
            self.dev_eng_size = len(self.dataset_loader.get_dev_data_eng())

        if args.dataset_name == 'csfd-allocine' or args.dataset_name == 'allocine-csfd' \
                or args.dataset_name == 'imdb-allocine' or args.dataset_name == 'allocine-imdb' \
                or args.dataset_name == 'sst-allocine' or args.dataset_name == 'allocine-sst':
            self.dev_data_target_loader = build_data_loader(self.dataset_loader.get_target_lang_dev_data(),
                                                            self.batch_size)

            self.dev_target_size = len(self.dataset_loader.get_target_lang_dev_data())
            self.target_language = self.dataset_loader.get_target_lang()
            self.source_language = self.dataset_loader.get_source_lang()

        logger.info("Train size:" + str(self.train_size))
        logger.info("Dev size:" + str(self.dev_size))
        logger.info("Test size:" + str(self.test_size))

        logger.info("Train labels distribution:")
        print_labels_distribution(self.dataset_loader.get_train_data(), "label_text", 'Train', args)

        logger.info(70 * "*")
        logger.info("Dev labels distribution:")
        print_labels_distribution(self.dataset_loader.get_dev_data(), "label_text", 'Dev', args)

        logger.info(70 * "*")
        logger.info("Test labels distribution:")
        print_labels_distribution(self.dataset_loader.get_test_data(), "label_text", 'Test', args)
        logger.info(70 * "*")

        if args.dataset_name == 'imdb-csfd':
            logger.info("Czech dev size:" + str(self.dev_czech_size))
        if args.dataset_name == 'csfd-imdb':
            logger.info("English dev size:" + str(self.dev_eng_size))
        if args.dataset_name == 'csfd-allocine' or args.dataset_name == 'allocine-csfd' \
                or args.dataset_name == 'imdb-allocine' or args.dataset_name == 'allocine-imdb' \
                or args.dataset_name == 'sst-allocine' or args.dataset_name == 'allocine-sst':
            logger.info(str(self.target_language) + " target dev size:" + str(self.dev_target_size))

        logger.info("Dataset loaded")
        logger.info(f"Number of labels in dataset:{self.num_labels}")

        self.prompt = self.build_prompt()
        logger.info("Built prompt:" + str(self.prompt))

    # Pozor model muze predikovat i neutral takze dostanu horsi vysledky, protoze kdyby predikoval
    # jenom binarne tak se splete mene casteji
    @staticmethod
    def get_predictions_from_file(file_path, classes: list[str], ignore_unknown: bool = False):
        predictions_df = pd.read_csv(file_path, sep='\t', header=0)
        if ignore_unknown:
            logger.info("Ignoring unknown classes, size before:" + str(len(predictions_df)))
            predictions_df = predictions_df[predictions_df.text_predicted.isin(classes)]
            predictions_df.reset_index(drop=True, inplace=True)
            logger.info("Size after filtering:" + str(len(predictions_df)))
        else:
            logger.info("Not ignoring unknown classes")

        review_texts = []
        predictions = []
        predictions_text = []
        real_values = []

        for index, row in predictions_df.iterrows():
            text = row['text_review']
            label = row['label']
            prediction = row['prediction']
            pred_text = row['text_predicted']

            predictions.append(prediction)
            review_texts.append(text)
            predictions_text.append(pred_text)
            real_values.append(label)

        return review_texts, predictions, predictions_text, real_values

    def get_predictions(self, data_loader):
        review_texts = []
        predictions = []
        predictions_text = []
        real_values = []

        failed_pred_texts = []

        class_names = self.dataset_loader.get_class_names()

        prediction_file = self.args.result_file + "_predictions.txt"
        logger.info("Writing results in file:" + str(prediction_file))

        with open(prediction_file, "w", encoding='utf-8', newline='') as f:

            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            writer.writerow(
                ["text_review", "label", "label_text", "prediction", "text_predicted", "text_predicted_orig"])
            f.flush()

            for i, data in enumerate(tqdm(data_loader)):
                if FAST_DEBUG_PREDICTIONS is True:
                    # only for testing purposes
                    if i == NUM_FAST_DEBUG_ITER:
                        break

                texts = data["text"]
                labels = data["labels"]

                for text, label in zip(texts, labels):
                    label = int(label)
                    pred_text = self.get_prediction(text)
                    orig_pred_text = pred_text
                    if pred_text is None:
                        logger.info("We were not able to get prediction for text:" + str(text))
                        failed_pred_texts.append(text)
                        continue
                    else:
                        if self.args.prompt_type == "advanced":
                            pred_text = self.extract_advanced_response(pred_text, class_names)

                        pred_text = pred_text.lower()
                        if pred_text not in class_names:
                            logger.info("Prediction is not one word for text:" + str(pred_text))
                            prediction = -1
                        else:
                            prediction = self.dataset_loader.get_label4text(pred_text)
                    predictions.append(prediction)
                    review_texts.append(text)
                    predictions_text.append(pred_text)
                    real_values.append(label)
                    label_text = self.dataset_loader.get_text4label(label)
                    writer.writerow(
                        [str(text), str(label), label_text, str(prediction),
                         pred_text, str(orig_pred_text)])
                    f.flush()
                    if self.args.enable_wandb:
                        try:
                            wandb.log({"text": text, "label": label, "label_text": label_text, "prediction": prediction,
                                       "text_prediction": pred_text, "text_predicted_orig": orig_pred_text})
                        except Exception as e:
                            logger.error("Error during wandb logging", e)
                    time.sleep(self.sleep_time_s)

        logger.info("Number of text that we were not able to classify:" + str(len(failed_pred_texts)))
        if len(failed_pred_texts) > 0:
            failed_predictions_file = self.args.result_file + "_failed_texts.txt"
            logger.info("Writing failed examples into file:" + str(failed_predictions_file))
            try:
                with open(failed_predictions_file, "w", encoding='utf-8', newline='') as f_failed:
                    writer_failed = csv.writer(f_failed, delimiter='\t', lineterminator='\n')
                    writer_failed.writerow(["failed_text"])
                    f_failed.flush()

                    for failed_text in failed_pred_texts:
                        writer.writerow([str(failed_text)])
            except Exception as e:
                logger.error("Error during writing failed examples:" + str(e))

        return review_texts, predictions, predictions_text, real_values

    def perform_evaluation(self):
        t0 = time.time()
        if self.args.reeval_file_path is not None:
            if self.args.reeval_ignore_unknown_classes:
                results_file = self.args.reeval_file_path + "_reeval_ignore_unknown_classes.txt"
            else:
                results_file = self.args.reeval_file_path + "_reeval.txt"

            classes = self.dataset_loader.get_class_names()

            review_texts, y_pred, predictions_text, y_test = \
                self.get_predictions_from_file(self.args.reeval_file_path,
                                               classes, self.args.reeval_ignore_unknown_classes)
        else:
            results_file = self.args.result_file
            review_texts, y_pred, predictions_text, y_test = self.get_predictions(self.test_data_loader)

        eval_time = time.time() - t0
        print_time_info(eval_time, len(y_pred), "Test results", file=None)

        clas_report = classification_report(y_test, y_pred, target_names=self.dataset_loader.get_class_names(),
                                            labels=self.dataset_loader.get_classes()
                                            )

        logger.info("\n" + clas_report)

        f1, accuracy, precision, recall = evaluate_predictions(y_pred, y_test)
        f1_micro, _, precision_micro, recall_micro = evaluate_predictions(y_pred, y_test, average='micro')

        result_string, only_results = get_table_result_string(
            f'{self.args.dataset_name}\tTransformer test:{self.args.model_name} {self.args}',
            f1, accuracy, precision, recall, eval_time,
            f1_micro=f1_micro, prec_micro=precision_micro, rec_mic=recall_micro)

        if self.args.enable_wandb is True:
            try:
                wandb.run.summary['f1'] = f1
                wandb.run.summary['accuracy'] = accuracy
                wandb.run.summary['precision'] = precision
                wandb.run.summary['recall'] = recall

                wandb.run.summary['f1_micro'] = f1_micro
                wandb.run.summary['precision_micro'] = precision_micro
                wandb.run.summary['recall_micro'] = recall_micro

                wandb.run.summary['class_report'] = str(clas_report)
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

        result_string = "\n-----------Test Results------------\n\t" + result_string

        logger.info("\n\n\n-----------Save results------------\n" + str(only_results) + "\n\n\n")

        with open(results_file, "w", encoding='utf-8') as f:
            f.write(only_results + "\n")

        if self.args.enable_wandb is True:
            try:
                wandb.run.summary['results_string'] = only_results
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

        logger.info(result_string)

        if self.args.enable_wandb is True:
            try:
                wandb.run.summary['result_string_head'] = result_string
                wandb.join()
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

    @abstractmethod
    def get_prediction(self, text: str) -> str:
        pass

    @abstractmethod
    def build_advanced_prompt(self):
        pass

    @abstractmethod
    def build_basic_prompt(self):
        pass

    @abstractmethod
    def build_in_context_prompt(self):
        pass

    def _sample_train_example(self):
        num_examples = self.args.in_context_num_examples
        train_data: pd.DataFrame = self.dataset_loader.get_train_data()

        sampled_data_df = pd.DataFrame()

        classes = self.dataset_loader.get_class_names()
        for i in range(0, num_examples):
            class_index = i % len(classes)
            clazz = classes[class_index]
            df_classes = train_data[train_data['label_text'] == clazz]
            random_example = df_classes.sample(1, random_state=RANDOM_SEED)

            sampled_data_df = pd.concat([sampled_data_df, random_example])

            # we want to get always the same result for each dataset
            train_data = train_data.sample(frac=1, random_state=i).reset_index(drop=True)

        # sampled_data = train_data.sample(num_examples, random_state=RANDOM_SEED)
        # sampled_data.reset_index(drop=True, inplace=True)
        return sampled_data_df

    def build_prompt(self):
        if self.args.prompt_type == "basic":
            prompt = self.build_basic_prompt()
        elif self.args.prompt_type == "advanced":
            prompt = self.build_advanced_prompt()
        elif self.args.prompt_type == "in_context":
            prompt = self.build_in_context_prompt()
        else:
            raise Exception("Unknown prompt type:" + str(self.args.prompt_type))

        return prompt

    @staticmethod
    def extract_advanced_response(text: str, class_names: list) -> str:
        text = text.split(PROMPT_DELIMITER)[-1].strip()
        text = text.lower()
        if text not in class_names:
            if text.endswith("negative"):
                text = "negative"
            elif text.endswith("positive"):
                text = "positive"
            elif text.endswith("neutral"):
                text = "neutral"
        return text

    def build_in_context_examples(self):
        samples: pd.DataFrame = self._sample_train_example()
        examples_prompt = []

        template = IN_CONTEXT_EXAMPLE_TEMPLATE

        for index, row in samples.iterrows():
            text = row['text']
            label_text = row['label_text']
            tmp = template.format(example=text, sentiment=label_text)
            examples_prompt.append(tmp)

        examples = "\n".join(examples_prompt) + "\n"
        return examples


class ChatGPTChatClassifier(ChatClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.user_prompt = self.build_user_prompt_part()
        # https://platform.openai.com/docs/api-reference/completions/create
        # Simillar to LLama model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.top_p = args.top_p

        logger.info("Loading credentials for ChatGPT")
        credentials_path = DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT
        if self.args.credentials_file_path is not None:
            credentials_path = self.args.credentials_file_path

        api_key = load_credentials_open_ai(credentials_path)
        init_open_ai(api_key)

    def get_prediction(self, text: str) -> str:
        num_attempts = 0
        while True:
            try:
                num_attempts += 1
                result = classify_sentiment_chatgpt(self.args.model_version,
                                                    self.prompt,
                                                    self.user_prompt,
                                                    text,
                                                    not self.disable_creating_new_conversation,
                                                    True,
                                                    temperature=self.temperature,
                                                    max_tokens=self.max_tokens,
                                                    top_p=self.top_p)
            except Exception as e:
                logger.info("Error:" + str(e))
                logger.info(f"Sleep for {self.sleep_time_retry_attempt_s} secs...")
                time.sleep(self.sleep_time_retry_attempt_s)
                result = None

            if result is not None or num_attempts > self.max_retry_attempts:
                break

        return result

    def build_user_prompt_part(self):
        if self.args.prompt_type == "basic":
            user_prompt = "The review:\n\n{text}"
        elif self.args.prompt_type == "advanced":
            user_prompt = "\n\n////{text}////"
        elif self.args.prompt_type == "in_context":
            user_prompt = "The review:\n\n{text}"
        else:
            raise Exception("Unknown prompt type:" + str(self.args.prompt_type))

        return user_prompt

    def build_basic_prompt(self):
        if self.args.binary:
            prompt = BASIC_BINARY_PROMPT
        else:
            prompt = BASIC_PROMPT

        return prompt

    def build_advanced_prompt(self):
        if self.args.binary:
            prompt = ADVANCED_PROMPT_BINARY
        else:
            prompt = ADVANCED_PROMPT

        return prompt

    def build_in_context_prompt(self):
        examples = self.build_in_context_examples()

        if self.args.binary:
            prompt = IN_CONTEXT_LEARNING_PROMPT_BINARY
        else:
            prompt = IN_CONTEXT_LEARNING_PROMPT

        prompt = prompt.format(examples=examples)
        return prompt


class LLamaChatClassifier(ChatClassifier):
    def __init__(self, args):
        super().__init__(args)
        self.max_retry_attempts = 100000
        self.sleep_time_retry_attempt_s = 60

        logger.info("Loading credentials for LLama 2")
        credentials_path = DEFAULT_CREDENTIALS_FILE_PATH_LLAMA
        if self.args.credentials_file_path is not None:
            credentials_path = self.args.credentials_file_path

        username, pwd = load_credentials(credentials_path)
        self.chatbot = init_hugging_chat(username, pwd)

    def get_prediction(self, text: str) -> str:
        num_attempts = 0
        while True:
            try:
                num_attempts += 1
                result = classify_sentiment(self.chatbot, self.prompt, text,
                                            not self.disable_creating_new_conversation, True)
            except Exception as e:
                logger.info("Error:" + str(e))
                logger.info(f"Sleep for {self.sleep_time_retry_attempt_s} secs...")
                time.sleep(self.sleep_time_retry_attempt_s)
                result = None

            if result is not None or num_attempts > self.max_retry_attempts:
                break

        return result

    def build_basic_prompt(self):
        if self.args.binary:
            prompt = BASIC_BINARY_PROMPT
        else:
            prompt = BASIC_PROMPT

        prompt = prompt + "The review:\n\n{text}"
        return prompt

    def build_advanced_prompt(self):
        if self.args.binary:
            prompt = ADVANCED_PROMPT_BINARY
        else:
            prompt = ADVANCED_PROMPT

        prompt = prompt + "\n\n////{text}////"
        return prompt

    def build_in_context_prompt(self):
        examples = self.build_in_context_examples()

        if self.args.binary:
            prompt = IN_CONTEXT_LEARNING_PROMPT_BINARY
        else:
            prompt = IN_CONTEXT_LEARNING_PROMPT

        prompt = prompt.format(examples=examples)

        prompt = prompt + "The review:\n\n{text}"
        return prompt
