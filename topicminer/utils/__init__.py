"""
Utilities Toolkit for TopicMiner

topicminer/utils/__init__.py

This module initializes the utilities package for the TopicMiner project, consolidating and providing direct access to
common functions used across various parts of the project. The utilities include functions for date conversion, text processing,
email handling, and interactive widgets, making them essential for processing and analyzing email data efficiently.
"""

from .utils import (
    convert_date_format,
    parse_date_day_time,
    prepare_corpus_and_dictionary,
)

from .text_processing import (
    read_text_file,
    preprocess_text,
    view_file 
)

from .email_processing import (
    parse_top_email_from_chain,
    load_unwanted_email_text,
    clean_email_body_text,
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