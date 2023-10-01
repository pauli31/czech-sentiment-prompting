import datetime
from collections import Counter
from pathlib import Path
import os
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import LOG_DIR, LOGGING_FORMAT, LOGGING_DATE_FORMAT

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)


def get_actual_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H.%M_%S.%f")


def print_time_info(time_sec, total_examples, label, per_examples=1000, print_fce=logger.info,
                    file=None):
    formatted_time = format_time(time_sec)
    time_per_examples = time_sec / total_examples * per_examples
    print_fce(100 * '$$$$$$')
    print_fce("Times for:" + str(label))
    print_fce(f'Examples:{total_examples}')
    print_fce(f'Total time for {str(label)} format: {formatted_time}')
    print_fce(f'Total time for {str(label)} in sec: {time_sec}')
    print_fce('----')
    print_fce(f'Total time for {str(per_examples)} examples format: {format_time(time_per_examples)}')
    print_fce(f'Total time for {str(per_examples)} examples in sec: {time_per_examples}')
    print_fce('----')
    print_fce('Copy ')
    # label | per_examples | total_examples | formatted_time | time_sec | time_per_examples
    output = str(label) + '\t' + str(per_examples) + '\t' + str(total_examples) + '\t' + str(formatted_time) + \
             '\t' + str(time_sec) + '\t' + str(time_per_examples)
    print_fce(output)
    # write results to disk
    if file is not None:
        file_write = file + "_" + label + ".txt"
        with open(file_write, 'a', encoding='utf-8') as f:
            f.write(output + "\n")

    print_fce(100 * '$$$$$$')

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_file_name(config, f1=None):
    time = get_actual_time()
    model_name = Path(config['model_name']).name
    binary = config['binary']
    model_ver = config['model_version']

    max_test_data = config['max_test_data']
    if max_test_data > 1:
        max_test_data = int(max_test_data)

    dataset_name = config['dataset_name']

    name_file = model_name + "_" \
                + model_ver + "_" \
                + dataset_name \
                + "_BIN-" + str(binary) \
                + "_prompt-" + str(config['prompt_type']) \
                + "_MXT-" + str(max_test_data)
    # + "_BS-" + str(batch_size) \
    # + "_EC-" + str(epochs) \
    # + "_LEN-" + str(config['max_seq_len']) \
    # + "_LR-%.7f" % (config['lr']) \
    # + "_SCH-" + str(config['lr_scheduler_name']) \
    # + "_WR-" + str(config['warm_up_steps']) \
    # + "_Opt-" + config['optimizer'] \
    # + "_CPU-" + str(config['use_cpu']) \
    # + "_TRAIN-" + str(config['use_only_train_data']) \
    # + "_WD-%.5f" % config['weight_decay'] \
    # + "_TWE-" + str(config['trainable_word_embeddings']) \
    # + "_MW-" + str(config['max_words']) \
    # + "_DR-" + str(config['dropout_rnn']) \
    # + "_DF-" + str(config['dropout_final'])


    name_file += "_" + time
    name_file = name_file.replace('.', '-')
    if f1 is not None:
        name_file = name_file + "_F1-%.4f" % f1

    return name_file


def generate_file_name_transformer(config):
    """
    If the full_mode is True, it returns dictionary otherwise it returns only one file

    :param config:
    :return:
    """
    time = get_actual_time()
    # epochs = config['epoch_num']
    binary = config['binary']
    # batch_size = config['batch_size']
    model_name = Path(config['model_name']).name
    # full_mode = config['full_mode']
    max_test_data = config['max_test_data']
    if max_test_data > 1:
        max_test_data = int(max_test_data)

    dataset_name = config['dataset_name']
    # if dataset_name == 'combined':
    #     tmp = '-'.join(config['combined_datasets'])
    #     dataset_name = dataset_name + '-' + tmp


    # if config['eval'] is True:
    #     model_name = model_name[:30]

    # num_iter = 1
    # if full_mode is True:
    #     num_iter = epochs
    model_ver = config['model_version']

    name_files = {}
    for i in range(1, 2):
        # if we are in full mode we change the epochs
        # if full_mode is True:
        #     epochs = i

        name_file = model_name + "_" \
                    + model_ver + "_" \
                    + dataset_name \
                    + "_BIN-" + str(binary) \
                    + "_prompt-" + str(config['prompt_type']) \
                    + "_MXT-" + str(max_test_data)
        # + "_FRZ-" + str(config['freze_base_model']) \
        # + "_BS-" + str(batch_size) \
        # + "_EC-" + str(epochs) \
        # + "_LR-%.7f" % (config['lr']) \
        # + "_LEN-" + str(config['max_seq_len']) \
        # + "_SCH-" + str(config['scheduler']) \
        # + "_TRN-" + str(config['use_only_train_data']) \
        # + "_F-" + str(full_mode)

        name_file += "_" + time
        name_file = name_file.replace('.', '-')
        name_files[i] = name_file

    # if full_mode is False:
    #     name_files = name_file


    return name_files

def init_loging(args, parser, result_file, generating_fce=generate_file_name_transformer, set_format=True):

    config_name = generating_fce(vars(args))
    file_name = os.path.join(LOG_DIR, str(config_name) + '.log')
    parser.add_argument("--config_name",
                        default=config_name)
    parser.add_argument("--result_file",
                        default=result_file)
    args = parser.parse_args()

    if set_format:
        # just to reset logging settings
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format=LOGGING_FORMAT,
                            datefmt=LOGGING_DATE_FORMAT,
                            filename=file_name)

        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)


        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logging.root.setLevel(level=logging.INFO)
        if args.silent is True:
            # logging.root.setLevel(level=logging.ERROR)
            console_handler.setLevel(level=logging.ERROR)
        else:
            # logging.root.setLevel(level=logging.INFO)
            console_handler.setLevel(level=logging.INFO)

        logging.getLogger().addHandler(console_handler)

    return args


def print_labels_distribution(data_df, label_col, desc, args):

    if data_df is None:
        logger.info("Data df are None")
        return

    # if args.classification_mode == MULTI_LABEL_MODE:
    #     labels = data_df[label_col].to_numpy().tolist()
    #     labels = [item for sublist in labels for item in sublist]
    #     counts = Counter(labels)
    # elif args.classification_mode == MULTI_CLASS_MODE:
    counts = Counter(data_df[label_col].to_numpy())
    # else:
    #     raise Exception("Unknown mode:" + str(args.classification_mode))

    # sort it
    # most = counts.most_common(100)
    most = counts.most_common()
    logger.info(desc + str(most))

    numbers = [100, 50, 20, 10, 1]
    at_least = []
    less = []
    for number in numbers:
        tmp_at_least, tmp_less = compute_occurences(number, most)
        at_least.append(tmp_at_least)
        less.append(tmp_less)


    # for i, number in enumerate(numbers):
    #     tmp_at_least = at_least[i]
    #     tmp_less = less[i]
    #
    #     logger.info(f'Have At least {number}:{tmp_at_least}')
    #     logger.info(f'Not At least {number}:{tmp_less}')
    #     logger.info("--")

def get_table_result_string(config_string, mean_f1_macro, mean_acc, mean_prec, mean_recall,
                            train_test_time, f1_micro, prec_micro, rec_mic):
    results = f'{config_string}\t{mean_f1_macro}\t{mean_acc}\t{mean_prec}\t{mean_recall}\t{f1_micro}\t{prec_micro}\t{rec_mic}\t{int(train_test_time)} s'
    results_head = '\tF1 Macro\tAccuracy\tPrecision\tRecall\tF1 Micro\tPrecision Micro\tRecall Micro\ttime\n' + results

    return results_head, results

def evaluate_predictions(y_pred, y_test, average='macro'):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)

    return f1, accuracy, precision, recall

def compute_occurences(number, most):
    total_size = len(most)
    at_least = [x for x in most if x[1] >= number]
    less = [x for x in most if x[1] < number]

    # logger.info(f'There are labels with at least {number}: total:{total_size} have:{len(at_least)} not:{len(less)}')

    return at_least, less

def load_wandb_api_key(path, default_value='fail'):
    try:
        with open(path, "r", encoding='utf-8') as f:
            data = f.read().replace('\n', '')
            data = data.strip()
    except Exception as e:
        if default_value == 'fail':
            raise e
        else:
            data = default_value
    return data