"""
Utilities Toolkit for TopicMiner

This module initializes the utilities package for the TopicMiner project, consolidating and providing direct access to
common functions used across various parts of the project. The utilities include functions for date conversion, text processing,
email handling, and interactive widgets, making them essential for processing and analyzing email data efficiently.

Available Imports:
- convert_date_format: Converts dates from 'YYYY-MM-DD' to 'Month DD, YYYY'.
- parse_date_day_time: Parses string representations of dates and times into structured formats.
- prepare_corpus_and_dictionary: Prepares a dictionary and corpus for topic modeling from email data.
- read_text_file: Reads text files while handling different encodings.
- preprocess_text: Cleans and standardizes text by applying various text preprocessing techniques.
- parse_top_email_from_chain: Parses the most recent email from a chain of threaded emails.
- load_unwanted_email_text: Loads a list of unwanted text phrases from a JSON file.
- clean_text_email_body: Cleans an email body by removing unwanted text phrases.
- add_unwanted_email_text: Adds a new unwanted text phrase to a JSON file.
- delete_unwanted_email_text: Removes an unwanted text phrase from a JSON file.
- email_viewer: Displays an email in a formatted HTML table within Jupyter Notebooks.
- email_viewer_widget: Creates an interactive widget for viewing and managing email documents in Jupyter Notebooks.

The functions in this module are designed to work seamlessly with the data structures and formats typically used in
the TopicMiner project, ensuring a smooth workflow for users and developers alike.
"""
from .utils import (
    convert_date_format,
    parse_date_day_time,
    prepare_corpus_and_dictionary,
)

from .text_processing import (
    read_text_file,
    preprocess_text,
)

from .email_processing import (
    parse_top_email_from_chain,
    load_unwanted_email_text,
    clean_text_email_body,
    add_unwanted_email_text,
    delete_unwanted_email_text,
)

from .widgets import (
    email_viewer,
    email_viewer_widget,
)

from .statistical_transforms import(
    create_tfidf_corpus
)