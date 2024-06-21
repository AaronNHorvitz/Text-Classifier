# TODO: Update email processing functions to include Dask and batch processing for larger loads.

"""
-------------------------------------------------------------------------------
File: email_processing.py
Written by: Aaron Noah Horvitz
Last Revision Date: 6/10/2024

Description:
-----------
This module is part of the Topic Miner project, aimed at processing and analyzing
emails saved as .txt files. It provides essential functions for reading, formatting,
cleaning, and parsing email data, making it ready for analysis and modeling. This
module is particularly useful in preprocessing steps where emails need to be extracted
from their raw formats, cleaned of unwanted text, and formatted for further analytical
processing.

Key Functions:
-------------
- generate_email_text: Generates structured email text from provided components.
- add_unwanted_email_text: Adds new unwanted phrases to a JSON file for text cleaning.
- delete_unwanted_email_text: Removes specified unwanted phrases from a JSON file.
- load_unwanted_email_text: Loads unwanted phrases from a JSON file for text preprocessing.
- clean_text_email_body: Cleans the email body by removing specified unwanted text.
- format_and_save_emails: Formats and saves emails from a DataFrame to text files.
- parse_top_email_from_chain: Extracts the top email from a chain of emails.
- process_emails_in_directory: Processes all emails within a directory into a DataFrame for analysis.

Dependencies:
------------
- Standard Library: json, os, re, textwrap
- Third-party Libraries: pandas, nltk
- Data Files: JSON files for unwanted texts, email text files for processing

Usage:
-----
This module is designed to be imported and used in Python scripts or Jupyter Notebooks
where comprehensive email data processing is required. It supports tasks ranging from
simple email text generation to complex processing of large sets of email data stored in directories.

Examples:
--------
To process emails in a directory and load their cleaned data into a DataFrame:
>>> data_path = './data/emails/'
>>> unwanted_text_file_path = './data/unwanted_texts.json'
>>> emails_df = process_emails_in_directory(data_path, unwanted_text_file_path)
>>> print(emails_df.head())

-------------------------------------------------------------------------------
"""


# Standard library imports
import chardet
import os
import re
import json
import textwrap
import logging
from datetime import datetime

# Data processing and mathematical operations
import numpy as np
import pandas as pd

# Natural Language Processing (NLP) tools
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import data

# data.path.append('/projects/merc_text_analytics/nltk_data') # Add the path to the NLTK data directory if the data is not found in local
data.path.append("./topicminoer/data/nltk_data")

# Utilities for progress tracking
from tqdm import tqdm

from ..config import (
    RAW_DATA_DIR,
    UNWANTED_TEXTS_FILE,
    PROCESSED_DATA_DIR_CSV,
    PROCESSED_DATA_DIR_JSON,
)


def generate_email_text(
    email_date: str,
    email_recipient: str,
    email_from: str,
    email_subject: str,
    email_body: str,
    word_wrap_limit: int = 100,
) -> str:
    """
    Generates formatted email text with word-wrapped body. The function formats the provided email details into
    a standard email structure with properly wrapped text and standardized placeholders for missing information.

    Parameters
    ----------
    email_date : str
        Date of the document. If the date is invalid or NaN, it will default to "(Unknown Date)".
    email_recipient : str
        Recipient of the email. If this field is NaN, it defaults to "(Unknown Recipient)".
    email_from : str
        Sender of the email. If this field is NaN, it defaults to "(Unknown Sender)".
    email_subject : str
        Subject of the email. If this field is NaN or explicitly '(NO SUBJECT)', it defaults to "(No Subject)".
    email_body : str
        Body of the email which will be formatted to ensure each line does not exceed 100 characters and breaks
        are at the end of the last word that fits within this limit.

    Returns
    -------
    str
        A string that represents the formatted email text, structured and aligned as an actual email would be.

    Examples
    --------
    >>> generate_email_text(
        email_date='2011-01-13',
        email_recipient='John Doe',
        email_from='Jane Doe',
        email_subject='Meeting Reminder',
        email_body='Please remember to attend the meeting scheduled at 10 AM tomorrow.'
    )
    """
    email_date = (
        convert_date_format(email_date) if not pd.isna(email_date) else "(Unknown Date)"
    )
    email_from = email_from if not pd.isna(email_from) else "(Unknown Sender)"
    email_recipient = (
        email_recipient if not pd.isna(email_recipient) else "(Unknown Recipient)"
    )

    # Correctly handle no subject and unknown subject cases
    if pd.isna(email_subject) or email_subject.strip() == "(NO SUBJECT)":
        email_subject = "(No Subject)"
    else:
        email_subject = email_subject.strip()

    # Ensure the email body is wrapped correctly
    email_body = textwrap.fill(email_body, width=word_wrap_limit)

    # Construct the email string
    email_str = f"""
To: {email_recipient}
From: {email_from}
Sent: {email_date}
Subject: {email_subject}
 
{email_body}
    """
    return email_str.strip()


def add_unwanted_email_text(
    new_text: str, file_path: str = "./data/unwanted_texts/unwanted_texts.json"
) -> None:
    """
    Adds a new unwanted text string to a JSON file. If the file does not exist, this function
    creates a new one with the given text. It also ensures that duplicates are not added.

    Parameters:
    ----------
    new_text : str
        New text string to add to the file.
    file_path : str, optional
        Path to the JSON file storing unwanted texts. Default is './data/unwanted_texts/unwanted_texts.json'.

    Returns:
    -------
    None

    Examples:
    --------
    >>> add_unwanted_email_text(new_text="Example unwanted text")
    Text added successfully.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Try to read the existing contents of the file
    try:
        with open(file_path, "r") as file:
            unwanted_texts = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Initialize the list if the file doesn't exist or if there's a decode error
        unwanted_texts = []

    # Add the new text if it's not already in the list
    if new_text not in unwanted_texts:
        unwanted_texts.append(new_text)
        with open(file_path, "w") as file:
            json.dump(unwanted_texts, file, indent=4)

    print("Text added successfully.")


def delete_unwanted_email_text(
    text_to_delete: str, file_path: str = "./data/unwanted_texts/unwanted_texts.json"
) -> None:
    """
    Deletes an unwanted text string from a JSON file. If the file does not exist or is empty,
    it notifies the user. It also handles the removal operation safely by checking the presence
    of the text to delete.

    Parameters:
    ----------
    text_to_delete : str
        Text string to delete from the file.
    file_path : str, optional
        Path to the JSON file storing unwanted texts. Default is './data/unwanted_texts/unwanted_texts.json'.

    Returns:
    -------
    None

    Examples:
    --------
    >>> delete_unwanted_email_text(text_to_delete="Example unwanted text")
    Text removed successfully.
    """

    # Check if the file exists before attempting to open it
    if not os.path.exists(file_path):
        print("JSON file does not exist.")
        return

    try:
        with open(file_path, "r") as file:
            unwanted_texts = json.load(file)
    except json.JSONDecodeError:
        print("JSON file is empty or corrupted.")
        return

    # Remove the unwanted text if it exists in the list
    if text_to_delete in unwanted_texts:
        unwanted_texts.remove(text_to_delete)
        with open(file_path, "w") as file:
            json.dump(unwanted_texts, file, indent=4)
        print("Text removed successfully.")
    else:
        print("Text not found in the file.")


def format_and_save_emails(df: pd.DataFrame, output_dir: str):
    """
     Iterates through each row in a DataFrame, formats it as an email using predefined format,
     and saves each formatted email as a text file in the specified directory.

     Parameters
     ----------
     df : pd.DataFrame
         A pandas DataFrame containing email data with columns 'docDate', 'to', 'from', 'subject', 'docText', and 'docID'.
         Each row represents an email with details to be formatted into a text file.
     output_dir : str
         The path to the directory where the email text files will be saved. Each file is named using the 'docID'
         field from the DataFrame and saved with a '.txt' extension.

     Returns
     -------
     None
         This function does not return any value. It writes files directly to the filesystem.

     Examples
     --------
     Suppose `df_emails` is your DataFrame containing the email data, and '/path/to/output' is the directory where you want to save the emails:

     >>> format_and_save_emails(df_emails, '/path/to/output')
    Processed email document ID: C06245106
     Processed email document ID: C06245079
     Processed email document ID: C06245073
     ...and so on for each email in the DataFrame.

     Notes
     -----
     Ensure that the `output_dir` directory exists and is writable. This function will overwrite existing files without warning.
    """
    for index, row in df.iterrows():
        email_text = generate_email_text(
            row["docDate"], row["to"], row["from"], row["subject"], row["docText"]
        )
        file_path = f"{output_dir}/{row['docID']}.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(email_text)
        print(f"Processed email document ID: {row['docID']}")


def parse_top_email_from_chain(text_file_contents: list) -> tuple:
    """
    Parses the top email from a chain in a list of lines from an Outlook formatted email text file.

    Parameters
    ----------
    text_file_contents : list of str
        Lines from a text file containing an email chain. Each line corresponds to one line of the text file.

    Returns
    -------
    tuple
        Contains 'To', 'From', 'Cc', 'Sent date', 'Subject', and 'Body' of the top email as strings.
        Each component is extracted based on Outlook format patterns.

    Notes
    -----
    This function assumes a specific format where the most recent email appears first and is followed
    by previous emails, each introduced by a line stating "On [date] [sender] wrote:".
    """
    # Initialize default values for email components
    email_recipient = "Unknown Recipient"
    email_sender = "Unknown Sender"
    email_cc = "No CC"
    email_bcc = "No BCC"
    email_date = "Unknown Date"
    email_subject = "No Subject"
    email_attachments = "No Attachments"
    email_categories = "No Categories"
    email_body = []

    # Regex to detect start of an old email in the chain
    old_email_start = re.compile(r"^On .* wrote:$")

    # Start processing the first email
    for line in text_file_contents:
        stripped_line = line.strip()
        if old_email_start.match(stripped_line):
            break  # Stop reading when an old email reply starts
        elif stripped_line.startswith("From:"):
            email_sender = stripped_line.replace("From:", "").strip()
        elif stripped_line.startswith("Sent:"):
            email_date = stripped_line.replace("Sent:", "").strip()
        elif stripped_line.startswith("To:"):
            email_recipient = stripped_line.replace("To:", "").strip()
        elif stripped_line.startswith("Cc:"):
            email_cc = stripped_line.replace("Cc:", "").strip()
        elif stripped_line.startswith("Bcc:"):
            email_bcc = stripped_line.replace("Bcc:", "").strip()
        elif stripped_line.startswith("Subject:"):
            email_subject = stripped_line.replace("Subject:", "").strip()
        elif stripped_line.startswith("Attachments:"):
            email_attachments = stripped_line.replace("Attachments:", "").strip()
        elif stripped_line.startswith("Categories:"):
            email_categories = stripped_line.replace("Categories:", "").strip()
        elif stripped_line:
            email_body.append(stripped_line)  # Collecting body text

    email_body = "\n".join(email_body)  # Join all body lines into a single string
    return (
        email_recipient,
        email_sender,
        email_cc,
        email_bcc,
        email_date,
        email_subject,
        email_attachments,
        email_categories,
        email_body,
    )


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


def load_unwanted_email_text():

    try:
        with open(UNWANTED_TEXTS_FILE, "r") as file:
            data = json.load(file)
        if "unwanted_texts" in data:
            return set(data["unwanted_texts"])
        else:
            raise ValueError(
                f"JSON file at {UNWANTED_TEXTS_FILE} is missing the 'unwanted_texts' key."
            )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return set()
    except ValueError as ve:
        print(ve)
        return set()


def save_unwanted_texts(unwanted_texts_filepath, texts):
    """Save the unwanted texts to a JSON file, ensuring data is converted from a set to a list."""
    try:
        with open(unwanted_texts_filepath, "w") as file:
            json.dump({"unwanted_texts": list(texts)}, file)
        print(f"Saved {len(texts)} texts to {unwanted_texts_filepath}")
    except Exception as e:
        print(f"Error saving texts: {e}")


def preprocess_text(
    text: str,
    unwanted_text_filepath=UNWANTED_TEXTS_FILE,
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
    unwanted_texts = load_unwanted_email_text()

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

    return preprocessed_text.lower()


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


def clean_email_body_text(email_body: str, unwanted_texts: list) -> str:
    """
    Cleans the email body by selectively removing unwanted text phrases and preserving essential formatting.

    Parameters:
    ----------
    email_body : str
        The original body of the email which may contain unwanted phrases.
    unwanted_texts : list
        List of unwanted text strings.

    Returns:
    -------
    str
        The cleaned email body with all unwanted texts removed while preserving new lines and necessary formatting.
    """

    # Case insensitive removal of each unwanted phrase from the email body
    cleaned_body = email_body
    for unwanted_text in unwanted_texts:
        cleaned_body = re.sub(
            re.escape(unwanted_text), "", cleaned_body, flags=re.IGNORECASE
        )

    # Replace multiple whitespace characters with a single space and preserve line breaks
    cleaned_body = re.sub(r"[^\S]+", " ", cleaned_body)

    # Trim spaces at the beginning and end of each line
    cleaned_body = re.sub(r"(?m)^\s+|\s+$", "", cleaned_body)

    return cleaned_body


def process_emails_to_csv(save_df=True, return_df=True) -> pd.DataFrame:
    """
    Process all email .txt files in the specified directory, extracting and preprocessing relevant components,
    and optionally save the results to a DataFrame saved as a CSV file.

    Parameters
    ----------
    save_df : bool, optional
        Flag to determine whether to save the DataFrame to a CSV file, by default True.

    Returns
    -------
    DataFrame
        A DataFrame containing the following columns: 'doc_id', 'date', 'time', 'day_of_week',
        'email_recipient', 'email_sender', 'email_cc', 'email_bcc', 'email_subject',
        'email_attachments', 'email_categories', 'email_body', 'processed_text', 'file_path'.
        Each row in the DataFrame represents an email with its respective details and processed text.

    Notes
    -----
    This function loops through each .txt file in the specified directory and performs the following:
    - Reads the content of the file.
    - Parses the email components such as recipient, sender, date, subject, and body.
    - Processes the email body by removing unwanted texts and applying text normalization and cleaning.
    - Collects all relevant information into a DataFrame for further analysis or processing.

    Examples
    --------
    >>> emails_df = process_emails_to_dataframe()
    >>> print(emails_df.head())

    Ensure that the `RAW_DATA_DIR` and `PROCESSED_DATA_DIR` are correctly set to point to valid directories
    on your system. The function uses `os.listdir` to iterate through the files, so make sure the
    path does not contain subdirectories with non-email files.

    The 'preprocess_text' function referenced should handle the removal of unwanted texts, tokenization,
    removal of stop words, and any other text normalization required.
    """

    email_data = []

    # Loop through each file in the directory
    for filename in tqdm(os.listdir(RAW_DATA_DIR)):

        # Construct a file path to each .txt file in the data directory
        file_path = os.path.join(RAW_DATA_DIR, filename)

        # Ensure the file is a .txt file
        if not file_path.endswith(".txt"):
            continue

        # Extract from_line, sent_line, subject_line, body
        file_contents = read_text_file(file_path)
        (
            email_recipient,
            email_sender,
            email_cc,
            email_bcc,
            email_date,
            email_subject,
            email_attachments,
            email_categories,
            email_body,
        ) = parse_top_email_from_chain(file_contents)

        # Extract date, time, day_of_week from sent_line
        date, time, day_of_week = parse_date_day_time(date_str=email_date)

        # Recreate Document ID
        doc_id = filename.strip(".txt")

        # Apply the preprocessing function to the 'Email Text' column
        processed_text = preprocess_text(email_body)

        # Append the extracted email components to a list
        email_data.append(
            [
                doc_id,
                date,
                time,
                day_of_week,
                email_recipient,
                email_sender,
                email_cc,
                email_bcc,
                email_subject,
                email_attachments,
                email_categories,
                email_body,
                processed_text,
                file_path,
            ]
        )

    # Create a DataFrame
    df_emails = pd.DataFrame(
        email_data,
        columns=[
            "doc_id",
            "date",
            "time",
            "day_of_week",
            "email_recipient",
            "email_sender",
            "email_cc",
            "email_bcc",
            "email_subject",
            "email_attachments",
            "email_categories",
            "email_body",
            "processed_text",
            "file_path",
        ],
    )
    # Save to a dataframe if true.
    if save_df:
        df_emails.to_csv(os.path.join(PROCESSED_DATA_DIR_CSV, "processed_emails.csv"))

    # Return a datfarme true.
    if return_df:
        return df_emails


def append_to_json(data):
    """
    Appends data to a JSON file in a newline-delimited format, creating the file if it does not exist.

    Parameters
    ----------
    data : list 
        List of dictionaries, each dictionary containing data from one email.
    """
    file_path = os.path.join(PROCESSED_DATA_DIR_JSON, 'processed_emails.json')
    try:
        with open(file_path, 'a') as file:
            for item in data:
                json.dump(item, file)
                file.write('\n')
        logging.info(f'Successfully appended {len(data)} records to {file_path}')
    except Exception as e:
        logging.error(f'Failed to append data to {file_path}: {e}')

    
def process_emails_to_json(chunk_size: int = 100):
    """
    Processes emails from text files stored in a specified directory, extracts relevant data, 
    and serializes them into JSON format in chunks. Each processed email's data is appended 
    to a JSON file. If the file does not exist, it is created. 

    Parameters
    ----------
    chunk_size : int, optional
        Number of email files to process before appending their data as a batch to the JSON file.
        The default value is 100, which helps manage memory usage and ensures data is not lost
        in case of a failure, by periodically saving it.

    Processes
    ---------
    1. Iterates over every text file in a predefined directory that stores raw email data.
    2. Reads and parses each file to extract email metadata and body.
    3. Processes the email body to clean and prepare text for further analysis or machine learning modeling.
    4. Constructs a dictionary for each email with all relevant data fields.
    5. Accumulates processed emails into a list.
    6. Appends the accumulated list to a JSON file every `chunk_size` emails or after the last email is processed.
       This step involves checking if the JSON file exists, creating it if it doesn't, or appending to it if it does.
    7. Logs each step's success or failure, including any file-specific errors, which aids in debugging and maintenance.

    Outputs
    -------
    None directly returned by the function, but data is written to a JSON file in the specified directory.
    This function is typically used when batch processing is required, and direct interaction or immediate
    response from the function is not necessary.

    Example
    -------
    >>> process_emails_to_json(chunk_size=100)  # Process and append every 100 emails to the JSON file.

    Notes
    -----
    - This function is designed to be run as part of a larger batch processing or ETL pipeline.
    - It uses global path configurations from a config file to manage directory paths and logging.
    - Proper error handling and logging are implemented to ensure any issues during processing are recorded,
      making it easier to monitor and troubleshoot.
    """
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')]
    email_data = []

    for i, filename in enumerate(tqdm(files), 1):
        try:
            file_path = os.path.join(RAW_DATA_DIR, filename)
            file_contents = read_text_file(file_path)
            email_parts = parse_top_email_from_chain(file_contents)
            processed_text = preprocess_text(email_parts['email_body'])
            
            email_data.append(
                {
                    "doc_id": filename.strip(".txt"),
                    "date": email_parts["date"],
                    "time": email_parts["time"],
                    "day_of_week": email_parts["day_of_week"],
                    "email_recipient": email_parts["email_recipient"],
                    "email_sender": email_parts["email_sender"],
                    "email_cc": email_parts["email_cc"],
                    "email_bcc": email_parts["email_bcc"],
                    "email_subject": email_parts["email_subject"],
                    "email_attachments": email_parts["email_attachments"],
                    "email_categories": email_parts["email_categories"],
                    "email_body": email_parts["email_body"],
                    "processed_text": processed_text,
                    "file_path": file_path,
                }
            )

            if i % chunk_size == 0 or i == len(files):
                append_to_json(email_data)
                email_data = []  # Clear the list after saving to JSON

        except Exception as e:
            logging.error(f'Error processing file {filename}: {e}')