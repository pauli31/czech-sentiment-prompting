from args import build_parser
import logging
import os
import wandb
import sys
import time

from chat_classifier import LLamaChatClassifier, ChatGPTChatClassifier
from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT, RESULTS_DIR, LOG_DIR, WANDB_DIR, WANDB_API_KEY_PATH, \
    WANDB_USER_NAME_PATH
from utils import generate_file_name, generate_file_name_transformer, init_loging, print_time_info, \
    evaluate_predictions, load_wandb_api_key

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

def main():
    wandb_api_key = load_wandb_api_key(WANDB_API_KEY_PATH)
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
    # set user from config
    user = load_wandb_api_key(WANDB_USER_NAME_PATH, 'common')

    # remove username if there was other specified and set the one from config
    # if there is a username
    if user is not None:
        sys.argv.extend(['--user', user])

    print("Hi")
    parser = build_parser()
    args = parser.parse_args()

    result_file = generate_file_name(vars(args))
    result_file = result_file + ".results"
    result_file = os.path.join(RESULTS_DIR, result_file)

    args = init_loging(args, parser, result_file, generating_fce=generate_file_name, set_format=True)

    if args.enable_wandb is True:
        wandb_tmp_name = str(args.config_name)
        try:
            wandb.init(project="llm-sentiment", name=wandb_tmp_name, config=vars(args),
                       dir=WANDB_DIR, entity='pauli31')
        except Exception as e:
            logger.error("Error WANDB with exception e:" + str(e))

    if args.model_name == "llama":
        classifier = LLamaChatClassifier(args)
    elif args.model_name == "chatgpt":
        classifier = ChatGPTChatClassifier(args)
    else:
        raise Exception("Unknown classifier:" + str(args.model_name))

    classifier.perform_evaluation()
    wandb.finish()
    print("Konec")




if __name__ == '__main__':
    # sys.argv.extend(['--dataset_name', 'sst'])
    # sys.argv.extend(['--model_name', 'llama'])
    # # sys.argv.extend(['--model_name', 'chatgpt'])
    # sys.argv.extend(['--batch_size', '1'])
    # sys.argv.extend(['--prompt_type', 'basic'])
    # sys.argv.extend(['--max_test_data', '500'])
    # # sys.argv.extend(['--enable_wandb'])
    # # sys.argv.extend(['--prompt_type', 'advanced'])
    # sys.argv.extend(['--prompt_type', 'in_context'])
    # # sys.argv.extend(['--binary'])
    # # sys.argv.extend(['--eval_binary_from_three_class_file', './results/llama_sst_BIN-False_2023-09-07_16-21_05-831390.results_predictions.txt'])

    main()



