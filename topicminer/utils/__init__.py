# topicminer/utils/__init__.py

"""
Utilities Toolkit for TopicMiner

This module initializes the utilities package, providing direct access to common
functions used across the TopicMiner project. It includes date conversion, text
processing, and other utility functions essential for processing and analyzing
email data.

Imports:
- convert_date_format: Converts dates from 'YYYY-MM-DD' to 'Month DD, YYYY'.
- parse_date_day_time: Parses string representations of date and time into structured format.
- read_text_file: Reads and processes text files, handling different encodings.
- preprocess_text: Applies cleaning and standardization procedures to text data.

"""

from .utils import convert_date_format, parse_date_day_time, prepare_corpus_and_dictionary
from .text_processing import read_text_file, preprocess_text


