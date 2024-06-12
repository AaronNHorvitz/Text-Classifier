
"""
-------------------------------------------------------------------------------
File: text_processing.py
Author: Aaron Noah Horvitz
Last Revision Date: 6/10/2024

Description:
-----------
Text Processing Module for TopicMiner

This module provides essential text processing utilities for the TopicMiner project, facilitating the cleaning and normalization of text data, particularly from email content. It is designed to enhance data readiness for NLP tasks such as tokenization, stop word removal, and lemmatization.


Features:
---------
- read_text_file: Reads text files while handling various encodings.
- preprocess_text: Performs comprehensive preprocessing on text data to prepare it for NLP modeling.

Dependencies:
-------------
- Standard Libraries: re
- Third-party Libraries: chardet, numpy, pandas, nltk
- TopicMiner Utilities: Functions from the 'email_processing' module for cleaning texts.

Usage:
------
These functions are intended for direct import into Python scripts or Jupyter Notebooks where text data needs extensive preprocessing before undergoing analysis or modeling.

Example:
--------
To preprocess text data from a file for NLP tasks:

>>> text_data = read_text_file('path/to/email.txt')
>>> cleaned_text = preprocess_text(text_data, 'path/to/unwanted_texts.json')
>>> print(cleaned_text)

Notes:
------
Ensure that the NLTK data path is correctly set if using NLTK resources for tokenization and lemmatization. This module assumes all text inputs are in English and makes use of English-specific processing such as stop word removal.
"""

# Standard library imports
import re

# Third-party library imports
import chardet
import numpy as np
import pandas as pd

# Natural Language Processing tools from NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import data

from topicminer.utils import clean_text_email_body

# data.path.append('/projects/merc_text_analytics/nltk_data') # Add the path to the NLTK data directory if the data is not found in local
data.path.append("./topicminoer/data/nltk_data")

def read_text_file(file_path: str, word_wrap_limit=100) -> list:
    """
    Read a text file and return its contents as a list of lines, automatically handling encoding detection.

    Parameters
    ----------
    file_path : str
        Path to the text file to be read.

    Returns
    -------
    list of str
        Lines read from the text file.

    Examples
    --------
    >>> file_contents = read_text_file('example.txt')
    >>> print(file_contents[0])  # Print the first line of the file.
    'This is the first line of the file.'

    Notes
    -----
    This function uses the `chardet` library to detect the encoding of the file,
    which helps in reading files with non-standard or mixed encodings.
    """
    # Uncover file encoding
    with open(file_path, "rb") as file:
        raw_data = file.read()
        encoding = chardet.detect(raw_data)["encoding"]  # Detect encoding

    # Read and return the contents of the file
    with open(file_path, "r", encoding=encoding) as file:
        text_file_contents = file.readlines()

    return text_file_contents


def preprocess_text(text: str, unwanted_texts_file_path: str) -> str:
    """
    Cleans and standardizes text by performing several preprocessing steps. This includes
    converting text to lowercase, removing specified unwanted phrases loaded from a JSON file,
    stripping out non-alphanumeric characters, removing stop words, and lemmatizing the remaining words.

    Parameters:
    ----------
    text : str
        The original text that needs to be preprocessed.
    unwanted_texts_file_path : str
        Path to the JSON file containing unwanted phrases to remove from the text.

    Returns:
    -------
    str
        The cleaned and processed text as a single string, with words normalized to their base form and separated by spaces.
    """
    # Load the list of unwanted texts from the specified JSON file
    unwanted_texts = load_unwanted_email_text(unwanted_texts_file_path)

    # Remove unwanted phrases from text
    text = clean_text_email_body(text, unwanted_texts)

    # Convert text to lowercase to standardize it
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Load English stopwords
    stop_words = set(stopwords.words("english"))

    # Filter out stopwords, non-alphabetic tokens, and single-character tokens
    tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words and len(word) > 1
    ]

    # Initialize the NLTK lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join words back into a single string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


