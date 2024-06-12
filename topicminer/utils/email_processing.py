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

import json
import os
import re
import textwrap

import pandas as pd
from tqdm import tqdm

from topicminer.utils import convert_date_format, parse_date_day_time, read_text_file, preprocess_text

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

def add_unwanted_email_text(new_text: str, file_path: str = './data/unwanted_texts/unwanted_texts.json') -> None:
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


def delete_unwanted_email_text(text_to_delete: str, file_path: str = './data/unwanted_texts/unwanted_texts.json') -> None:
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
        

def load_unwanted_email_text(file_path: str = './data/unwanted_texts/unwanted_texts.json') -> list:
    """
    Reads all unwanted text strings from a JSON file and returns them as a list. If the file does not exist,
    it creates an empty file at the default path and notifies the user, returning an empty list. If the file is empty or corrupted,
    it notifies the user and returns an empty list.

    Parameters
    ----------
    file_path : str, optional
        Path to the JSON file storing unwanted texts. Defaults to './data/unwanted_texts/unwanted_texts.json'.

    Returns
    -------
    list
        A list of unwanted text strings. Returns an empty list if the file is not found, is empty, or corrupted.

    Examples
    --------
    >>> load_unwanted_texts()  # Using default path
    []
    >>> load_unwanted_texts("./custom_path/to/unwanted_texts.json")
    ['Example unwanted text', 'Another unwanted phrase']

    Notes
    -----
    The function ensures that the directory for the file exists, creating it if necessary. It uses a relative path by default,
    which assumes the function is called from the root of the project directory.
    """
    # Ensure the directory exists (if not, create it)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file exists
    if not os.path.exists(file_path):
        print("JSON file does not exist. Creating an empty file.")
        with open(file_path, "w") as file:
            json.dump([], file)  # Create an empty JSON array
        return []

    try:
        with open(file_path, "r") as file:
            unwanted_texts = json.load(file)
            return unwanted_texts
    except json.JSONDecodeError:
        print("JSON file containing unwanted texts is empty or corrupted.")
        return []


def clean_text_email_body(email_body: str, unwanted_texts: list) -> str:
    """
    Cleans the email body by removing all unwanted text phrases defined in a JSON file.

    Parameters:
    ----------
    email_body : str
        The original body of the email which may contain unwanted phrases.
    unwanted_texts : list
        List of unwanted text strings.

    Returns:
    -------
    str
        The cleaned email body with all unwanted texts removed.
    """

    # Remove each unwanted phrase from the email body
    cleaned_body = email_body.lower()

    # Remove each unwanted phrase from the email body
    for unwanted_text in unwanted_texts:
        cleaned_body = cleaned_body.replace(unwanted_text.lower(), "")

        # Remove unwanted symbols except for alphanumeric and spaces, handle new lines and carriage returns
        pattern = r"[^\w\s]|[\r\n]"
        cleaned_body = re.sub(pattern, lambda x: ' ' if x.group(0) in '\n\r' else '', cleaned_body, flags=re.UNICODE)
    
        # Collapse multiple whitespaces into a single space and trim leading/trailing spaces
        cleaned_body = re.sub(r'\s+', ' ', cleaned_body).strip()

    return cleaned_body

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

def process_emails_in_directory(data_path: str) -> pd.DataFrame:
    """
    Processes all email .txt files in the specified directory,
    extracting and preprocessing relevant components, and returns a DataFrame.

    Parameters
    ----------
    data_path : str
        Path to the directory containing email .txt files.
    unwanted_text_file_path : str
        Path to the JSON file containing unwanted text phrases that should be removed
        during the preprocessing of emails.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns: 'doc_id', 'date', 'time', 'day_of_week',
        'to_line', 'from_line', 'subj_line', 'email_body', 'processed_text', and 'file_path'.
        Each row in the DataFrame represents an email with its respective details and processed text.

    Examples
    --------
    >>> data_path = './data/emails/'
    >>> unwanted_text_file_path = './data/unwanted_texts.json'
    >>> emails_df = process_emails_in_directory(data_path, unwanted_text_file_path)
    >>> print(emails_df.head())

    Notes
    -----
    This function loops through each .txt file in the specified directory and performs the following:
    - Reads the content of the file.
    - Parses the email components such as recipient, sender, date, subject, and body.
    - Processes the email body by removing unwanted texts and applying text normalization and cleaning.
    - Collects all relevant information into a DataFrame for further analysis or processing.

    The 'preprocess_text' function referenced in this code should handle the removal of unwanted texts,
    tokenization, removal of stop words, and any other text normalization required.

    Ensure that the `data_path` and `unwanted_text_file_path` are correctly set to point to valid directories
    and files on your system. The function uses `os.listdir` to iterate through the files, so make sure the
    path does not contain subdirectories with non-email files.
    """

    email_data = []

    # Loop through each file in the directory
    for filename in tqdm(os.listdir(data_path)):

        # Construct a file path to each .txt file in the data directory
        file_path = os.path.join(data_path, filename)

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
    return df_emails
