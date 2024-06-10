"""
-------------------------------------------------------------------------------
File: utils.py
Written by: Aaron Noah Horvitz
Last Revision Date: 6/10/2024

Description:
-----------
This module is part of the TopicMiner project, designed to process and analyze
emails saved as .txt files. It includes essential functions for converting and parsing
date formats, and preprocessing email data for further analysis and modeling. These utilities
are crucial for the standardization and readability improvements needed for effective
topic modeling and other NLP tasks.

Key Functions:
-------------
- convert_date_format: Converts dates from 'YYYY-MM-DD' to 'Month DD, YYYY'.
- parse_date_day_time: Parses dates and times into a structured format, returning
  standardized outputs.
- prepare_corpus_and_dictionary: Prepares a dictionary and corpus for topic modeling.

Dependencies:
------------
- Python Standard Library: datetime
- Third-party Libraries: numpy, pandas, gensim
- Data Files: N/A

Usage:
-----
This module is intended for direct import into Python scripts or Jupyter Notebooks
where comprehensive preprocessing of email data is required. It simplifies the handling
of date formats commonly found in email headers and prepares text data for NLP operations.

-------------------------------------------------------------------------------
"""

# Standard library imports
from datetime import datetime
from typing import Tuple, List

# Third-party imports
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from nltk import data

# Append NLTK data directory path
data.path.append("./topicminer/data/nltk_data")

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



def prepare_corpus_and_dictionary(
    emails_df: pd.DataFrame,
    processed_text_col: str = "processed_text",
    no_below: int = 5,  # Good starting point for moderate-sized corpora
    no_above: float = 0.5,  # Exclude words appearing in more than 50% of the documents
) -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
    """
    Prepares a dictionary and corpus from the provided DataFrame column for use in topic modeling,
    with options to filter tokens by document frequency.

    Parameters
    ----------
    emails_df : pd.DataFrame
        DataFrame containing the emails.
    processed_text_col : str
        Column name in `emails_df` which contains preprocessed text for topic modeling.
    no_below : int
        Minimum number of documents a token must appear in to be kept.
    no_above : float
        Maximum proportion of documents a token can appear in to be kept.

    Returns
    -------
    tuple
        A tuple containing:
        - Dictionary: A Gensim Dictionary object of the unique tokens in the texts.
        - list: A list of bag-of-words (BoW) tuples for each document in the texts.

    Raises
    ------
    ValueError
        If `processed_text_col` is not a column in `emails_df`.
    """
    if processed_text_col not in emails_df.columns:
        raise ValueError(
            f"Column {processed_text_col} does not exist in the DataFrame."
        )

    # Prepare texts and document IDs
    texts = [doc.split() for doc in emails_df[processed_text_col]]

    # Create a Gensim Dictionary object
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # Create a bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus

