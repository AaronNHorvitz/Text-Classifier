import os
import logging

# Absolute path to the root of the project (one level up from the directory containing the config.py file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'topicminer.log')

# Path definitions for various resources
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR_CSV = os.path.join(DATA_DIR, 'processed_data', 'csv_format')
PROCESSED_DATA_DIR_JSON = os.path.join(DATA_DIR, 'processed_data','json_format')
UNWANTED_TEXTS_FILE = os.path.join(DATA_DIR, 'unwanted_texts', 'unwanted_texts.json')

# Example configuration for logging
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'topicminer.log')

# Ensure that essential directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR_CSV, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR_JSON, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='a'
)

# To allow logging to console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)