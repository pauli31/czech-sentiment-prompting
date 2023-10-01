import os
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, 'data')
POLARITY_DIR = os.path.join(DATA_DIR, 'polarity')
LOG_DIR = os.path.join(BASE_PATH, 'logs')
TENSOR_BOARD_LOGS = os.path.join(LOG_DIR, 'tensor-logs')
RESULTS_DIR = os.path.join(BASE_PATH, "results")
WANDB_DIR = os.path.join(BASE_PATH, 'wandb')
DEFAULT_CREDENTIALS_FILE_PATH_LLAMA = os.path.join(BASE_PATH, "private", "credentials.txt")
DEFAULT_CREDENTIALS_FILE_PATH_CHATGPT = os.path.join(BASE_PATH, "private", "credentials_chatgpt.txt")

WANDB_API_KEY_PATH = os.path.join(BASE_PATH, 'private', 'wandb_api_key.txt')
WANDB_USER_NAME_PATH = os.path.join(BASE_PATH, 'private', 'user.txt')


Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(TENSOR_BOARD_LOGS).mkdir(parents=True, exist_ok=True)
Path(WANDB_DIR).mkdir(parents=True, exist_ok=True)


# fb dataset dirs
FACEBOOK_DATASET_DIR = os.path.join(POLARITY_DIR, 'fb', 'splitted')
FACEBOOK_DATASET_TRAIN = os.path.join(FACEBOOK_DATASET_DIR, 'train', 'train.csv')
FACEBOOK_DATASET_TEST = os.path.join(FACEBOOK_DATASET_DIR, 'test', 'test.csv')
FACEBOOK_DATASET_DEV = os.path.join(FACEBOOK_DATASET_DIR, 'dev', 'dev.csv')
FACEBOOK_DATASET = os.path.join(FACEBOOK_DATASET_DIR, 'dataset.csv')

# csfd dataset dirs
CSFD_DATASET_DIR = os.path.join(POLARITY_DIR, 'csfd', 'splitted')
CSFD_DATASET_TRAIN = os.path.join(CSFD_DATASET_DIR, 'train', 'train.csv')
CSFD_DATASET_TEST = os.path.join(CSFD_DATASET_DIR, 'test', 'test.csv')
CSFD_DATASET_DEV = os.path.join(CSFD_DATASET_DIR, 'dev', 'dev.csv')
CSFD_DATASET = os.path.join(CSFD_DATASET_DIR, 'dataset.csv')

# mallcz cz dataset dirs
MALL_DATASET_DIR = os.path.join(POLARITY_DIR, 'mallcz', 'splitted')
MALL_DATASET_TRAIN = os.path.join(MALL_DATASET_DIR, 'train', 'train.csv')
MALL_DATASET_TEST = os.path.join(MALL_DATASET_DIR, 'test', 'test.csv')
MALL_DATASET_DEV = os.path.join(MALL_DATASET_DIR, 'dev', 'dev.csv')
MALL_DATASET = os.path.join(MALL_DATASET_DIR, 'dataset.csv')

IMDB_DATASET_CL_DIR = os.path.join(POLARITY_DIR, 'imdb', 'splitted-cl')
IMDB_DATASET_CL_TRAIN = os.path.join(IMDB_DATASET_CL_DIR, 'train', 'train.csv')
IMDB_DATASET_CL_TEST = os.path.join(IMDB_DATASET_CL_DIR, 'test', 'test.csv')
IMDB_DATASET_CL_DEV = os.path.join(IMDB_DATASET_CL_DIR, 'dev', 'dev.csv')
IMDB_DATASET_CL = os.path.join(IMDB_DATASET_CL_DIR, 'dataset.csv')

IMDB_DATASET_DIR = os.path.join(POLARITY_DIR, 'imdb', 'splitted')
IMDB_DATASET_TRAIN = os.path.join(IMDB_DATASET_DIR, 'train', 'train.csv')
IMDB_DATASET_TEST = os.path.join(IMDB_DATASET_DIR, 'test', 'test.csv')
IMDB_DATASET_DEV = os.path.join(IMDB_DATASET_DIR, 'dev', 'dev.csv')
IMDB_DATASET = os.path.join(IMDB_DATASET_DIR, 'dataset.csv')


# SST_DATASET_DIR = os.path.join(POLARITY_DIR, 'sst', 'od-jakuba')
# SST_DATASET_TRAIN = os.path.join(SST_DATASET_DIR, 'sst_train.csv')
# SST_DATASET_TEST = os.path.join(SST_DATASET_DIR, 'sst_test.csv')
# SST_DATASET_DEV = os.path.join(SST_DATASET_DIR, 'sst_dev.csv')
# SST_DATASET = os.path.join(SST_DATASET_DIR, 'sst_entire_dataset.csv')

SST_DATASET_DIR = os.path.join(POLARITY_DIR, 'sst', 'split')
SST_DATASET_TRAIN = os.path.join(SST_DATASET_DIR, 'train', 'train.csv')
SST_DATASET_TEST = os.path.join(SST_DATASET_DIR, 'test', 'test.csv')
SST_DATASET_DEV = os.path.join(SST_DATASET_DIR, 'dev', 'dev.csv')
SST_DATASET = os.path.join(SST_DATASET_DIR, 'dataset.csv')


ALLOCINE_DATASET_DIR = os.path.join(POLARITY_DIR, 'allocine', 'split')
ALLOCINE_DATASET_TRAIN = os.path.join(ALLOCINE_DATASET_DIR, 'train', 'train.csv')
ALLOCINE_DATASET_TEST = os.path.join(ALLOCINE_DATASET_DIR, 'test', 'test.csv')
ALLOCINE_DATASET_DEV = os.path.join(ALLOCINE_DATASET_DIR, 'dev', 'dev.csv')
ALLOCINE_DATASET = os.path.join(ALLOCINE_DATASET_DIR, 'dataset.csv')

# combined
COMBINED_DATASET_DIR = os.path.join(POLARITY_DIR, 'combined')
Path(COMBINED_DATASET_DIR).mkdir(parents=True, exist_ok=True)


# corresponds to dir names in POLARITY_DIR
DATASET_NAMES = ['fb', 'csfd', 'sst', 'mallcz','imdb', 'allocine', 'combined', 'imdb-csfd', 'csfd-imdb', 'sst-csfd',
                 'csfd-sst', 'allocine-csfd', 'csfd-allocine', 'imdb-allocine', 'allocine-imdb', 'sst-allocine', 'allocine-sst']

# all real datasets without combined
DATASET_NAMES_COMBINED = ['fb', 'csfd', 'mallcz']

LOGGING_FORMAT= '%(asctime)s: %(levelname)s: %(name)s %(message)s'
LOGGING_DATE_FORMAT = '%m/%d/%Y %H:%M:%S'


# Important to reproduce results
RANDOM_SEED = 666




