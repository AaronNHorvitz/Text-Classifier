import os
import logging

# Absolute path to the root of the project (one level up from the directory containing the config.py file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Path definitions for various resources
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR_CSV = os.path.join(DATA_DIR, 'processed_data', 'csv_format')
PROCESSED_DATA_DIR_JSON = os.path.join(DATA_DIR, 'processed_data', 'json_format')
UNWANTED_TEXTS_FILE = os.path.join(DATA_DIR, 'unwanted_texts', 'unwanted_texts.json')

# Logging configuration
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
console_handler.setLevel(logging.CRITICAL)  # No logs will appear in the notebook unless critical
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# Default column name for processed text in the data
PROCESSED_TEXT_COL = "processed_text"

# Configuration for topic modeling filtering
TOKEN_FILTER_NO_BELOW = 2  # Tokens must appear in at least this many documents
TOKEN_FILTER_NO_ABOVE = 0.5  # Tokens must appear in no more than 10% of documents

# Configuration guidance for TOKEN_FILTER_NO_BELOW and TOKEN_FILTER_NO_ABOVE
# These parameters control the filtering of tokens in the dictionary creation process based on their document frequency.
# Adjust these settings based on the number of documents in your corpus to improve model relevance and performance.

# Small datasets (e.g., hundreds of documents):
# - TOKEN_FILTER_NO_BELOW = 2  # Keep tokens that appear in at least 2 documents
# - TOKEN_FILTER_NO_ABOVE = 0.5  # Keep tokens that appear in no more than 50% of the documents

# Medium datasets (e.g., thousands of documents):
# - TOKEN_FILTER_NO_BELOW = 5  # Keep tokens that appear in at least 5 documents
# - TOKEN_FILTER_NO_ABOVE = 0.3  # Keep tokens that appear in no more than 30% of the documents

# Large datasets (e.g., tens of thousands of documents):
# - TOKEN_FILTER_NO_BELOW = 20  # Keep tokens that appear in at least 20 documents
# - TOKEN_FILTER_NO_ABOVE = 0.1  # Keep tokens that appear in no more than 10% of the documents

# Very large datasets (e.g., hundreds of thousands of documents):
# - TOKEN_FILTER_NO_BELOW = 50  # Keep tokens that appear in at least 50 documents
# - TOKEN_FILTER_NO_ABOVE = 0.05  # Keep tokens that appear in no more than 5% of the documents

# These thresholds are initially set for a dataset size of approximately 10K-100K documents.
# TOKEN_FILTER_NO_BELOW = 20  # Example setting
# TOKEN_FILTER_NO_ABOVE = 0.1  # Example setting

# Adjust the above values according to the specific characteristics and requirements of your dataset.

