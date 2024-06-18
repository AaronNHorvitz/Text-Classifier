
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

# # Append appropriate pathways to sys
# import sys
# import os

# src_path = os.path.dirname(os.getcwd()) # Find home directory
# path_to_append = os.path.join(src_path, 'utils') # Append utils
# sys.path.append(path_to_append)
# sys.path.append(src_path)

# Standard library imports
from datetime import datetime

# Third-party library imports
import pandas as pd
import numpy as np
import chardet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Local application imports
from topicminer.utils.email_processing import clean_email_body_text, load_unwanted_email_text

# data.path.append('/projects/merc_text_analytics/nltk_data') # Add the path to the NLTK data directory if the data is not found in local
#data.path.append("./topicminoer/data/nltk_data")

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


def preprocess_text(
        text: str,
        unwanted_text_file_path="topicminer/data/unwanted_texts.json",
        ) -> str:
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
    unwanted_texts = load_unwanted_email_text(unwanted_text_file_path)

    # Remove unwanted phrases from text and remove unwanted white spaces and symbols
    text = clean_email_body_text(text, unwanted_texts)

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

    # Initialize the NLTK lemmatizer and lemmatize the words. 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join words back into a single string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text

def view_file(text_file_path):
    """
    Read and return the content of a text file, automatically detecting and applying the correct text encoding.

    The function opens a file in binary mode to first read raw data for encoding detection via the `chardet` library.
    It then reopens the file with the detected encoding to read and return the text content. This ensures that the file
    is read correctly according to its character encoding, which is crucial for accurately processing text data in various
    formats and encodings.

    Parameters
    ----------
    text_file_path : str
        The file path for the text file to be read.

    Returns
    -------
    str
        The content of the file as a string.

    Examples
    --------
    >>> text_content = view_file('example.txt')
    >>> print(text_content)

    Notes
    -----
    The function uses the `chardet` library to detect encoding, which can handle a variety of text encodings but
    may not always be 100% accurate for every file type or content.
    """
    # Uncover file encoding
    with open(text_file_path, "rb") as file:
        raw_data = file.read()

        # Use chardet library to detect encoding in the text file
        encoding = chardet.detect(raw_data)["encoding"]

    # Read and print file
    with open(text_file_path, "r", encoding=encoding) as file:
        return file.read()
    

def convert_date_format(date_str: str) -> str:
    """
    Convert a date string from 'YYYY-MM-DD' format to 'Month DD, YYYY' format.

    This function takes a date string in the ISO 8601 date format (YYYY-MM-DD) and converts
    it to a more readable string format that includes the full month name, the day, and the year.

    Parameters:
    ----------
    date_str : str
        A string representing the date in 'YYYY-MM-DD' format. This should be a valid date string
        that conforms to the ISO 8601 date format.

    Returns:
    -------
    str
        A string representing the date in 'Month DD, YYYY' format. If `date_str` is NaN or an invalid
        date, the function will return NaN.

    Examples:
    --------
    >>> convert_date_format('2011-01-13')
    'January 13, 2011'

    Notes
    -----
    The function uses the `pandas.isna()` function to check for NaN values and `datetime.strptime` from
    Python's datetime module to parse and format the date. If an invalid date is provided that doesn't match
    the 'YYYY-MM-DD' format, it will raise a ValueError.
    """
    # Check if the date string is NaN
    if pd.isna(date_str):
        return np.nan

    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Format the datetime object into the desired string format
    new_date_str = date_obj.strftime("%B %d, %Y")

    return new_date_str

def parse_date_day_time(date_str):
    """
     Parses a string representing a date and time, returning the day of the week, date, and time in a standardized format.

     This function is designed to handle multiple date formats by checking each format defined in `date_formats`. It uses
     the `datetime.strptime` method to try parsing the date string according to each format until successful. If none of
     the formats match, or the date string is a known placeholder for missing data, it defaults to returning "Unknown" for
     each part of the date.

     Parameters
     ----------
     date_str : str
         The date string to parse. This can be in various formats or a known placeholder indicating missing data.

     Returns
     -------
     tuple
         A tuple containing three strings: (date, time, day_of_week). If parsing fails or the input is a known placeholder,
         these will return as "Unknown".

    Examples
     --------
     >>> parse_date_day_time("Monday, January 01, 2020 02:30 PM")
     ('January 01, 2020', '14:30', 'Monday')

     >>> parse_date_day_time("2020-01-01")
     ('January 01, 2020', 'Unknown', 'Unknown')

     >>> parse_date_day_time("Unknown Date")
     ('Unknown', 'Unknown', 'Unknown')

     Notes
     -----
     The function can handle custom formats by modifying the `date_formats` list. It uses Python's datetime library for
     parsing and formatting. If new formats are expected, they should be added to the `date_formats` list to ensure
     proper parsing.
    """
    # Define your date formats
    date_formats = [
        "%A, %B %d, %Y %I:%M %p",  # Original format
        "%Y-%m-%d",  # New expected format
    ]

    # Placeholder for unknown date
    unknown_placeholder = "Unknown Date"

    # Initialize default values for date, time, and day_of_week
    date = "Unknown"
    time = "Unknown"
    day_of_week = "Unknown"

    if date_str == unknown_placeholder or not date_str:
        return date, time, day_of_week

    for format in date_formats:
        try:
            # Attempt to parse the date
            date_obj = datetime.strptime(date_str, format)
            date = date_obj.strftime("%B %d, %Y")
            time = date_obj.strftime("%H:%M")  # 24 hour clock
            day_of_week = date_obj.strftime("%A")
            break  # Break the loop if the date is successfully parsed
        except ValueError:
            # If parsing fails, try the next format
            continue

    return date, time, day_of_week
