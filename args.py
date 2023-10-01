import argparse

from config import DATASET_NAMES


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Required parameters
    parser.add_argument("--dataset_name",
                        required=True,
                        choices=DATASET_NAMES,
                        help="The dataset that will be used, they correspond to names of folders")

    parser.add_argument("--prompt_type",
                        required=True,
                        choices=['basic', 'advanced', 'in_context'],
                        help="Type of prompt that will be used")

    parser.add_argument("--in_context_num_examples",
                        type=int,
                        default=4,
                        help="Number of examples if in context prompt is used"
                        )

    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        choices=['llama','chatgpt'],
                        help="Name of model ")

    # https://platform.openai.com/docs/models/gpt-3-5
    # gpt-3.5-turbo-0301 - snapshot from March 1st 2023
    # gpt-3.5-turbo - most recent model
    parser.add_argument("--model_version",
                        type=str,
                        default="gpt-3.5-turbo",
                        help="Only applicable to chatgpt model for now.")

    parser.add_argument("--credentials_file_path",
                        default=None,
                        type=str,
                        help="Path to the credentials file for HuggingFace Chat or ChatGPT"
                             " if not use default path './private/credentials.txt' is used instead for LLama model"
                             "and './private/credentials_chatgpt.txt for ChatGPT")

    parser.add_argument("--binary",
                        default=False,
                        action='store_true',
                        help="If used the polarity task is treated as binary classification, i.e., positve/negative"
                             " The neutral examples are dropped")

    parser.add_argument("--silent",
                        default=False,
                        action='store_true',
                        help="If used, logging is set to ERROR level, otherwise INFO is used")

    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Batch size")

    parser.add_argument("--use_only_train_data",
                        default=False,
                        action='store_true',
                        help="If set, the program will use training and development data for training,"
                             " i.e. it will use"
                             "train + dev for training, no validation is done during training")

    parser.add_argument("--disable_creating_new_conversation",
                        default=False,
                        action='store_true',
                        help="If used, no new conversation is used in the chatbot")

    parser.add_argument("--max_train_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for training, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")

    parser.add_argument("--max_test_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for testing, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")

    parser.add_argument("--temperature",
                        default=0.9,
                        type=float,
                        help="Temperature of the model, how much is creative, currently only applicable to ChatGPT")

    parser.add_argument("--top_p",
                        default=0.95,
                        type=float,
                        help="top_p parameter of the model, currently only applicable to ChatGPT")

    parser.add_argument("--sleep_time_s",
                        default=10,
                        type=float,
                        help="Sleep time after each request")

    parser.add_argument("--max_tokens",
                        default=1024,
                        type=int,
                        help="max_tokens parameter of the model, currently only applicable to ChatGPT")

    parser.add_argument("--reeval_file_path",
                        type=str,
                        default=None,
                        help="Path to file with *results_predictions, If argument is passed,"
                             " the predictions are loaded and evaluated again, otherwise the predictions are created"
                        )

    parser.add_argument("--reeval_ignore_unknown_classes",
                        default=False,
                        action='store_true',
                        help="If set, the unknown classes are ignored during reevaluation")

    parser.add_argument("--enable_wandb",
                        default=False,
                        action='store_true',
                        help="If set, the program will use wandb for logging, otherwise not")

    parser.add_argument('--user',
                        required=True,
                        choices=['Pauli', 'Jakub', 'Adam', 'common'],
                        help='User that runs the experiment')

    return parser
