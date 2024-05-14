# TODO: Eliminate redundancies in the script.
# TODO: Isolate modeling components to Utils.py.
# TODO: Ensure consistency with dependencies and functions in Utils.py.
# TODO: Verify and update type annotations throughout the script.
# TODO: Complete and edit existing docstrings.
# TODO: Coordinate with system administrators to install Black for code formatting.

"""
-------------------------------------------------------------------------------
File: tmbase.py
Written by: Aaron Noah Horvitz
Date: 4/18/2024
-------------------------------------------------------------------------------

This module, tmbase.py, is designed to analyze a collection of emails by performing 
natural language processing (NLP) and topic modeling. It supports extracting and 
transforming email data, performing side-by-side comparisons of original and processed
texts, and applying advanced modeling techniques such as Latent Dirichlet Allocation (LDA) 
and Doc2Vec to categorize and derive insights from the email corpus.

Features:
- Extract and transform email data for preprocessing.
- Compare original and transformed texts to verify integrity.
- Apply LDA for topic modeling and insights derivation.
- Utilize Doc2Vec for document categorization.

The script is structured to be modular, allowing specific components of the workflow
to be executed independently. It is optimized for use within a Jupyter Notebook environment,
providing visual feedback for analysis steps.

Dependencies:
- pandas for data manipulation.
- numpy for numerical operations.
- gensim for NLP and topic modeling.
- nltk for text processing.
- scikit-learn for applying clustering algorithms.
- matplotlib and IPython for visualization.

Usage:
This script can be executed in parts or as a whole in a Jupyter Notebook to provide
an interactive analysis experience. Functions are documented with specifics on parameters,
expected inputs, and outputs.

"""

# Standard library imports
import chardet
import os
import re
import json
from datetime import datetime

# Data processing and mathematical operations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# Visualization
import matplotlib.pyplot as plt

# Natural Language Processing (NLP) tools
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import data

# data.path.append('/projects/merc_text_analytics/nltk_data') # Add the path to the NLTK data directory if the data is not found in local
data.path.append("./topicminoer/data/nltk_data")

# Machine Learning and topic modeling
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LdaMulticore, TfidfModel, Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Utilities for display and progress tracking
from IPython.display import display, HTML
from ipywidgets import Button, HBox, VBox, Output, Layout, Text, Label
from tqdm import tqdm
from typing import Tuple, List

unwanted_texts = [
    "This mailbox does not process replies. Do not reply to this message.",
    "This mailbox does not process replies. Do not reply to this message.",
    "Visit the IT Advisory Site for all advisories issued. If you have any feedback on this communication, please contact us at IT.Communications@irs.gov",
    "visit the it advisory site for all advisories issued. if you have any feedback on this \ncommunication, please contact us at it.communications@irs.gov.",
    "NOTE: Visit the IT Advisory Site for all advisories issued. If you have any feedback on this communication, please contact us at IT.Communications@irs.gov.",
    "IT.Communications@irs.gov",
    "866-743-5748",
    "option 2",
    "IRS",
    "Internal Revenue Service",
]


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


def text_pruner(data_path: str, unwanted_texts_filename: str = "unwanted_texts.json") -> VBox:
    """
    Initialize an interactive text pruning interface for cleaning up email data from a specified directory.

    The function loads a list of unwanted text phrases from a JSON file and provides an interactive Jupyter
    widget interface. Users can navigate through text files, add or remove unwanted text phrases, and visually
    compare the original, cleaned, and processed versions of each text file.

    Parameters
    ----------
    data_path : str or os.PathLike
        The path to the directory containing the text files to be processed.
    unwanted_texts_filename : str, default 'unwanted_texts.json'
        The filename of the JSON file containing a list of unwanted text phrases.

    Returns
    -------
    VBox
        An ipywidgets VBox object containing all interactive widgets for navigating and editing texts.

    Examples
    --------
    >>> from pathlib import Path
    >>> text_pruner_interface = text_pruner(Path("/path/to/email/directory"))
    >>> display(text_pruner_interface)

    Notes
    -----
    The unwanted texts are initially loaded from a JSON file but can be dynamically modified through the interface.
    Each email text file should be UTF-8 encoded and have a '.txt' extension. The comparison and editing actions
    are immediately updated in the interface to reflect any changes.
    """
    # Create a directory for unwanted text files
    unwanted_text_dir = os.path.join(data_path, "unwanted_text")
    os.makedirs(unwanted_text_dir, exist_ok=True)
    unwanted_texts_file = os.path.join(unwanted_text_dir, unwanted_texts_filename)

    files = sorted([f for f in os.listdir(data_path) if f.endswith(".txt")])
    index = [0]

    # Define function to load unwanted texts
    def load_unwanted_texts(filepath):
        try:
            with open(filepath, "r") as file:
                return set(json.load(file).get("unwanted_texts", []))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    # Define function to save unwanted texts
    def save_unwanted_texts(filepath, texts):
        with open(filepath, "w") as file:
            json.dump({"unwanted_texts": list(texts)}, file)

    # Load unwanted texts
    unwanted_texts = load_unwanted_texts(unwanted_texts_file)

    # Widgets for interaction
    output = Output(layout={"border": "1px solid black", "width": "100%"})
    btn_prev = Button(description="Previous")
    btn_next = Button(description="Next")
    txt_add_unwanted = Text(placeholder="Add unwanted text")
    btn_add = Button(description="Add")
    txt_remove_unwanted = Text(placeholder="Remove unwanted text")
    btn_remove = Button(description="Remove")
    lbl_position = Label()

    def update_labels():
        lbl_position.value = f"Document {index[0] + 1} of {len(files)}"

    def show_email(idx):
        output.clear_output()
        file_path = os.path.join(data_path, files[idx])
        with open(file_path, "r", encoding="utf-8") as file:
            email_content = file.read()

        # Process the email content to remove unwanted texts
        email_content_no_unwanted = email_content
        for phrase in unwanted_texts:
            highlighted_phrase = f"<mark style='background-color: red;'>{phrase}</mark>"
            email_content = email_content.replace(phrase, highlighted_phrase)
            email_content_no_unwanted = email_content_no_unwanted.replace(phrase, "")

        # Simulate processing the email content
        processed_text = preprocess_email(email_content_no_unwanted)

        with output:
            display(
                HTML(
                    f"""
            <style>
                .email-view {{ width: 32%; overflow-wrap: break-word; white-space: pre-wrap; }}
                table {{ width: 100%; table-layout: fixed; }}
                td {{ vertical-align: top; }}
            </style>
            <table>
                <tr>
                    <td class="email-view"><b>Original Email (with highlights):</b><br>{email_content}</td>
                    <td class="email-view"><b>Email Without Unwanted Texts:</b><br>{email_content_no_unwanted}</td>
                    <td class="email-view"><b>Processed Email:</b><br>{processed_text}</td>
                </tr>
            </table>
            """
                )
            )
        update_labels()

    def preprocess_email(email_text):
        # Placeholder for email processing logic
        # For now, just replace new lines with spaces
        return " ".join(email_text.split())

    btn_prev.on_click(lambda b: navigate(-1))
    btn_next.on_click(lambda b: navigate(1))
    btn_add.on_click(lambda b: add_text(txt_add_unwanted.value))
    btn_remove.on_click(lambda b: remove_text(txt_remove_unwanted.value))

    def navigate(direction):
        new_index = max(0, min(len(files) - 1, index[0] + direction))
        if new_index != index[0]:
            index[0] = new_index
            show_email(index[0])

    def add_text(text):
        if text:
            unwanted_texts.add(text)
            save_unwanted_texts(unwanted_texts_file, unwanted_texts)
            txt_add_unwanted.value = ""
            show_email(index[0])

    def remove_text(text):
        if text in unwanted_texts:
            unwanted_texts.remove(text)
            save_unwanted_texts(unwanted_texts_file, unwanted_texts)
            txt_remove_unwanted.value = ""
            show_email(index[0])

    show_email(index[0])
    controls = HBox(
        [
            btn_prev,
            lbl_position,
            btn_next,
            txt_add_unwanted,
            btn_add,
            txt_remove_unwanted,
            btn_remove,
        ]
    )
    return VBox([controls, output])


# You would typically call the function like this:
# tp = text_pruner('path_to_your_data_directory')
# display(tp)


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


def parse_email_components(file_path):
    """
    Parses an email file to extract the header components and the body of the email. This function is designed to
    handle variations in email formatting, including missing or out-of-order header lines.

    The function first determines the file encoding using the `chardet` library to properly handle the file's contents.
    It then reads the email file line by line, capturing the 'To', 'From', 'Sent', and 'Subject' lines if they exist.
    Once it encounters an empty line, it begins to capture the remainder of the file as the body of the email.

    Parameters
    ----------
    file_path : str
        The file path of the email text file to be parsed.

    Returns
    -------
    tuple
        A tuple containing five elements:
        - to_line (str): The recipient of the email. Returns "Unknown Recipient" if not found.
        - from_line (str): The sender of the email. Returns "Unknown Sender" if not found.
        - sent_line (str): The sending date of the email. Returns "Unknown Date" if not found.
        - subject_line (str): The subject of the email. Returns "No Subject" if not found.
        - body (str): The body text of the email, which may include newlines.

    Examples
    --------
    >>> parse_email_components("path/to/email.txt")
    ('John Doe', 'Jane Smith', 'January 01, 2020', 'New Year Greetings', 'Happy New Year!')

    Notes
    -----
    The function assumes that the email headers are separated from the body by at least one blank line.
    If the expected headers are not present, it defaults to predefined placeholder strings to indicate missing data.
    This function uses a robust error handling strategy to ensure that even if some parts of the email are missing,
    the parsing can continue and return as much information as available.

    Raises
    ------
    FileNotFoundError
        If the `file_path` does not correspond to a valid file.

    UnicodeDecodeError
        If the file's encoding is not correctly detected and an error occurs during file reading.
    """
    # Initialize default values for email components
    to_line = "Unknown Recipient"
    from_line = "Unknown Sender"
    sent_line = "Unknown Date"
    subject_line = "No Subject"
    body = ""

    # Uncover file encoding
    with open(file_path, "rb") as file:
        raw_data = file.read()
        # Use chardet library to detect encoding in the text file
        encoding = chardet.detect(raw_data)["encoding"]

    with open(file_path, "r", encoding=encoding) as file:
        # Read the file contents
        contents = file.readlines()

    # Process each line
    body_start_found = False
    for line in contents:
        stripped_line = line.strip()
        if stripped_line.startswith("To:"):
            to_line = stripped_line.replace("To:", "").strip()
        elif stripped_line.startswith("From:"):
            from_line = stripped_line.replace("From:", "").strip()
        elif stripped_line.startswith("Sent:"):
            sent_line = stripped_line.replace("Sent:", "").strip()
        elif stripped_line.startswith("Subject:"):
            subject_line = stripped_line.replace("Subject:", "").strip()
        elif stripped_line == "":
            body_start_found = True
        elif body_start_found:
            body += stripped_line + "\n"

    return to_line, from_line, sent_line, subject_line, body


def preprocess_text(text, unwanted_texts):
    """
    Cleans and standardizes text by performing several preprocessing steps. This includes
    converting text to lowercase, removing specified unwanted phrases, stripping out
    non-alphanumeric characters, removing stop words, and lemmatizing the remaining words.
    The function aims to prepare text data for further natural language processing or
    machine learning tasks.

    Parameters
    ----------
    text : str
        The original text that needs to be preprocessed.
    unwanted_texts : list of str
        A list of phrases that should be removed from the text. This could include
        automated email signatures or disclaimers that are irrelevant for analysis.

    Returns
    -------
    str
        The cleaned and processed text as a single string, with words normalized to their
        base form and separated by spaces.

    Examples
    --------
    >>> preprocess_text("Please DO NOT reply to this message. Thank you!",
                        ["Do not reply"])
    'please thank you'

    Notes
    -----
    The function uses the NLTK library for tokenizing and lemmatizing the text, and for
    accessing a list of English stop words. It ensures that only alphabetic characters
    are retained, eliminating numbers and punctuation.

    Raises
    ------
    TypeError
        If the inputs are not in the expected string or list format.
    """

    # Standardize
    text = text.lower()

    # Remove unwanted texts
    for unwanted in unwanted_texts:
        unwanted = unwanted.lower()
        text = text.replace(unwanted, "")

    # Remove unwanted symbols
    pattern = r"[^a-zA-Z0-9\s]"
    text = re.sub(pattern, "", text)
    text = text.replace("\n", " ").replace("\r", "")

    # Proceed with other preprocessing steps
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


def process_emails_in_directory(data_path: str) -> pd.DataFrame:
    """
    Processes all email .txt files in the specified directory by extracting and preprocessing
    relevant components from each file. The function generates a DataFrame containing detailed
    components of each email, including extracted metadata like date and time, as well as the
    preprocessed body text.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the email .txt files. This directory should only contain
        .txt files that are formatted as emails.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents an email, with columns including:
        - 'doc_id': Document ID derived from the filename.
        - 'date', 'time', 'day_of_week': Extracted and processed date and time information.
        - 'to_line': The recipient of the email.
        - 'from_line': The sender of the email.
        - 'subj_line': The subject line of the email.
        - 'body': The raw body text of the email.
        - 'processed_body_text': The body text after preprocessing.
        - 'file_path': The path to the original .txt file.

    Examples
    --------
    >>> data_path = './data/emails/'
    >>> df_emails = process_emails_in_directory(data_path)
    >>> print(df_emails.head())

    Notes
    -----
    This function requires the presence of helper functions such as `parse_email_components` and
    `parse_date_day_time` for extracting email components and date information respectively, and
    `preprocess_text` for text preprocessing. It assumes all files in the specified directory are
    properly formatted email text files.
    """
    email_data = []

    # Loop through each file in the directory
    for filename in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, filename)

        # Ensure the file is a .txt file
        if not file_path.endswith(".txt"):
            continue

        # Extract email components
        to_line, from_line, sent_line, subject_line, body = parse_email_components(
            file_path=file_path
        )

        # Extract date, time, day_of_week from sent_line
        date, time, day_of_week = parse_date_day_time(date_str=sent_line)

        # Recreate Document ID
        doc_id = filename[:-4]  # Removing the '.txt' extension

        # Preprocess the body text
        processed_body_text = preprocess_text(body)

        # Append the extracted email components to a list
        email_data.append(
            [
                doc_id,
                date,
                time,
                day_of_week,
                to_line,
                from_line,
                subject_line,
                body,
                processed_body_text,
                file_path,
            ]
        )

    # Create a DataFrame from the list of email data
    df_emails = pd.DataFrame(
        email_data,
        columns=[
            "doc_id",
            "date",
            "time",
            "day_of_week",
            "to_line",
            "from_line",
            "subj_line",
            "body",
            "processed_body_text",
            "file_path",
        ],
    )
    return df_emails


def process_emails(data_path: str) -> pd.DataFrame:
    """
    Processes all email .txt files in the specified directory by extracting and preprocessing
    relevant components such as sender, recipient, date, and subject. It appends the preprocessed
    subject line to the body text and returns a DataFrame containing detailed components and
    combined processed text for each email.

    Parameters:
    ----------
    data_path : str
        Path to the directory containing email .txt files. This directory should only contain .txt files
        that represent emails, with each filename corresponding to a unique document ID.

    Returns:
    -------
    pd.DataFrame
        A DataFrame where each row represents an email, with columns including:
        - 'doc_id': Document ID derived from the filename.
        - 'date', 'time', 'day_of_week': Extracted date information.
        - 'to_line': Recipient of the email.
        - 'from_line': Sender of the email.
        - 'subj_line': Subject of the email.
        - 'email_body': Original text of the email body.
        - 'processed_text': Combined processed subject and body text.
        - 'file_path': Full path to the email file.

    Examples:
    --------
    >>> data_path = './data/emails/'
    >>> df_emails = process_emails(data_path)
    >>> print(df_emails.head())

    Note:
    -----
    This function requires the presence of helper functions such as `parse_email_components`,
    `parse_date_day_time`, and `preprocess_text` for extracting and processing email components.
    It assumes that all email files are properly formatted and that necessary preprocessing functions
    are available and correctly implemented.
    """
    email_data = []

    # Loop through each file in the directory
    for filename in tqdm(os.listdir(data_path)):
        # Construct a file path to each .txt file in the data directory
        file_path = os.path.join(data_path, filename)

        # Ensure the file is a .txt file
        if not file_path.endswith(".txt"):
            continue

        # Extract email components
        to_line, from_line, sent_line, subject_line, email_body = (
            parse_email_components(file_path=file_path)
        )

        # Extract date, time, day_of_week from sent_line
        date, time, day_of_week = parse_date_day_time(date_str=sent_line)

        # Recreate Document ID
        doc_id = filename.strip(".txt")

        # Preprocess the email body text
        processed_email_body_text = preprocess_text(email_body, unwanted_texts)

        # Preprocess the subject line and concatenate with the processed body text
        processed_subject = preprocess_text(subject_line, unwanted_texts)
        all_processed_text = (
            processed_subject + " " + processed_email_body_text
        )  # Concatenate with a space

        # Append the extracted email components to a list
        email_data.append(
            [
                doc_id,
                date,
                time,
                day_of_week,
                to_line,
                from_line,
                subject_line,
                email_body,
                all_processed_text,
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
            "to_line",
            "from_line",
            "subj_line",
            "email_body",
            "processed_text",
            "file_path",
        ],
    )
    return df_emails


def retrieve_email_text(doc_id: str, data_path: str) -> str:
    """
    Retrieves the content of an email from a text file based on the document ID.

    Parameters:
    ----------
    doc_id : str
        Document ID corresponding to the email. This ID should match the filename (without '.txt') in the data path.
    data_path : str
        Path to the directory where email .txt files are stored. The path should end with a slash.

    Returns:
    -------
    str
        The full text content of the email file.

    Example:
    --------
    >>> email_text = retrieve_email_text("001", "/path/to/email/files/")
    >>> print(email_text[:100])  # Prints the first 100 characters of the email.
    """
    file_path = os.path.join(data_path, f"{doc_id}.txt")
    with open(file_path, "r") as file:
        email_text = file.read()
    return email_text


def insert_line_breaks(text: str, line_length: int = 75) -> str:
    """
    Inserts line breaks into a text string to enhance readability, breaking lines at specified intervals.

    Parameters:
    ----------
    text : str
        The text string to format with line breaks.
    line_length : int
        The maximum number of characters on a line before a break is inserted.

    Returns:
    -------
    str
        The modified text string with line breaks inserted at the specified interval.

    Example:
    --------
    >>> formatted_text = insert_line_breaks("This is a long text string that needs to be broken into multiple lines.", 10)
    >>> print(formatted_text)
    This is a
    long text
    string tha
    t needs to
    be broken
    into multi
    ple lines.
    """
    return "\n".join(
        text[i : i + line_length] for i in range(0, len(text), line_length)
    )


def compare_email_versions(doc_id: str, data_path: str, df_emails: pd.DataFrame):
    """
    Displays a side-by-side comparison of the original email, a reconstructed version from a DataFrame,
    and the processed text with readability enhancements.

    This function reads the original email text from a file, uses DataFrame entries to reconstruct the
    email, and formats the processed text with line breaks for improved readability. It displays these
    versions side by side in a formatted HTML table.

    Parameters:
    ----------
    doc_id : str
        Document ID corresponding to the email. This ID should match the filename (without '.txt') in the data path.
    data_path : str
        Path to the directory where email .txt files are stored. This path should include the filename with the '.txt' extension.
    df_emails : pd.DataFrame
        DataFrame containing the email components and processed text. It must include columns 'doc_id', 'to_line', 'from_line',
        'date', 'subj_line', 'email_body', and 'processed_text'.

    Example:
    --------
    >>> data_path = "/path/to/emails"
    >>> df_emails = pd.DataFrame({
        "doc_id": ["001", "002"],
        "to_line": ["John Doe", "Jane Smith"],
        "from_line": ["Jane Doe", "John Smith"],
        "date": ["2021-06-01", "2021-06-02"],
        "subj_line": ["Greetings", "Re: Meeting"],
        "email_body": ["Hello John, how are you?", "Dear Jane, please find the meeting agenda attached."],
        "processed_text": ["hello john how are you", "dear jane please find meeting agenda attached"]
    })
    >>> compare_email_versions("001", data_path, df_emails)

    Notes:
    -----
    This function utilizes the IPython display capabilities to render HTML content within Jupyter notebooks or other compatible interfaces.
    It is designed for use within these environments and may not function as intended outside of them.
    """
    # Retrieve the original email text
    original_email_text = retrieve_email_text(doc_id, data_path)

    # Find the row in the DataFrame corresponding to doc_id
    email_row = df_emails[df_emails["doc_id"] == doc_id].iloc[0]

    # Reconstruct the email from the DataFrame components
    reconstructed_email = (
        f"To: {email_row['to_line']}\n"
        f"From: {email_row['from_line']}\n"
        f"Sent: {email_row['date']}\n"
        f"Subject: {email_row['subj_line']}\n\n"
        f"{email_row['email_body']}"
    )

    # Get the processed text and insert line breaks for readability
    processed_text = insert_line_breaks(email_row["processed_text"], line_length=100)

    # Prepare HTML content for display
    comparison_html = f"""
    <table>
        <tr>
            <th>Original Email From File</th>
            <th>Reconstructed Email From Parsed Components</th>
            <th>Processed Text For Modeling</th>
        </tr>
        <tr>
            <td><pre>{original_email_text}</pre></td>
            <td><pre>{reconstructed_email}</pre></td>
            <td><pre>{processed_text}</pre></td>
        </tr>
    </table>
    """

    # Display the comparison
    display(HTML(comparison_html))


def prepare_texts(df: pd.DataFrame, column_name: str) -> List[List[str]]:
    """
    Tokenizes the text content of a specific column in a pandas DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the text data to be tokenized.
    column_name : str
        Name of the column containing text to be tokenized.

    Returns:
    -------
    List[List[str]]
        A list of lists where each sublist contains tokens from one document in the DataFrame.

    Example:
    --------
    >>> df = pd.DataFrame({'text': ['hello world', 'text data mining']})
    >>> prepare_texts(df, 'text')
    [['hello', 'world'], ['text', 'data', 'mining']]
    """
    return [doc.split() for doc in df[column_name]]


def create_dictionary_and_corpus(
    texts: List[List[str]], no_below: int = 10, no_above: float = 0.9
) -> Tuple[Dictionary, List[Tuple[int, int]]]:
    """
    Creates a Gensim dictionary and Bag of Words (BoW) corpus from tokenized texts and applies frequency filters.

    Parameters:
    ----------
    texts : List[List[str]]
        List of tokenized texts where each text is a list of tokens.
    no_below : int
        Minimum document frequency for tokens. Tokens appearing in fewer than 'no_below' documents are removed.
    no_above : float
        Maximum document frequency, as a fraction of the total corpus size, for tokens. Tokens appearing in more than
        'no_above' fraction of the documents are removed.

    Returns:
    -------
    Tuple[Dictionary, List[Tuple[int, int]]]
        A tuple where the first element is a Gensim dictionary of the processed corpus, and the second element is
        a list of BoW representations for each document.

    Example:
    --------
    >>> texts = [['hello', 'world'], ['text', 'data', 'mining']]
    >>> dictionary, corpus = create_dictionary_and_corpus(texts, no_below=1, no_above=0.5)
    >>> print(list(corpus))
    [[(0, 1), (1, 1)], [(2, 1), (3, 1), (4, 1)]]
    """
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus


def interactive_email_viewer(data_path: str):
    """
    Creates an interactive viewer to navigate and display emails from a specified directory. The viewer includes
    navigation buttons to move between emails, and a text input to jump directly to an email by its document ID.
    Emails are displayed in a side-by-side format showing both the original and processed versions.

    Parameters:
    ----------
    data_path : str
        Path to the directory containing email text files. Each file should be a '.txt' file representing an email.

    Returns:
    -------
    VBox
        An ipywidgets VBox object containing the navigation controls and the email display area. This can be used
        within a Jupyter Notebook to interactively browse through email documents.

    Example:
    --------
    >>> interactive_viewer = interactive_email_viewer('./emails')
    >>> display(interactive_viewer)

    Notes:
    -----
    The function assumes that each email file is named using a unique identifier (doc ID) and ends with the '.txt'
    extension. It uses ipywidgets for interactivity, allowing users to browse through emails using "Previous" and
    "Next" buttons or by entering a specific document ID. The email content is displayed with both its original
    and processed text side by side for comparison.

    Functions used within:
    - `view_file(file_path)` should return the raw text content of the file.
    - `parse_email_components(file_path)` should parse and return components like to, from, sent date, subject,
      and body from the email content.
    - `preprocess_text(text, unwanted_texts)` should process the text by removing or altering unwanted elements
      to prepare it for further analysis or display.
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".txt")])
    index = [0]  # Mutable object to keep track of the index in a closure

    # Widgets
    output = Output(layout={"border": "1px solid black", "width": "100%"})
    btn_prev = Button(
        description="Previous", layout=Layout(width="100px", height="30px")
    )
    btn_next = Button(description="Next", layout=Layout(width="100px", height="30px"))
    doc_id_input = Text(
        description="Go to ID:",
        placeholder="Enter Document ID",
        layout=Layout(width="200px"),
    )
    btn_go = Button(description="Go", layout=Layout(width="80px", height="30px"))
    lbl_position = Label()
    lbl_doc_id = Label()

    def update_labels():
        """Updates the labels showing the current position and document ID in the list."""
        lbl_position.value = f"Document {index[0] + 1} of {len(files)}"
        lbl_doc_id.value = (
            f"Doc ID: {files[index[0]][:-4]}"  # Remove '.txt' extension for display
        )

    def show_email(idx: int):
        """Displays the email content based on the current index."""
        output.clear_output()
        file_path = os.path.join(data_path, files[idx])
        original_text = view_file(file_path)
        to_line, from_line, sent_line, subject_line, body = parse_email_components(
            file_path
        )
        processed_text = preprocess_text(body, unwanted_texts)

        with output:
            display(
                HTML(
                    f"""
            <style>
                .email-view {{ width: 49%; overflow-wrap: break-word; white-space: pre-wrap; }}
                table {{ width: 100%; table-layout: fixed; }}
                td {{ vertical-align: top; }}
            </style>
            <table>
                <tr>
                    <td class="email-view"><b>Original Email:</b><br>{original_text}</td>
                    <td class="email-view"><b>Processed Email:</b><br>{'From: ' + from_line}<br>{'Sent: ' + sent_line}<br>
                    {'Subject: ' + subject_line}<br><br>{processed_text}</td>
                </tr>
            </table>
            """
                )
            )
        update_labels()

    def on_prev_clicked(b):
        if index[0] > 0:
            index[0] -= 1
            show_email(index[0])

    def on_next_clicked(b):
        if index[0] < len(files) - 1:
            index[0] += 1
            show_email(index[0])

    def on_go_clicked(b):
        try:
            target_id = doc_id_input.value.strip() + ".txt"
            target_index = files.index(target_id)
            index[0] = target_index
            show_email(index[0])
        except ValueError:
            output.clear_output()
            with output:
                print("Document ID not found!")

    # Button click events
    btn_prev.on_click(on_prev_clicked)
    btn_next.on_click(on_next_clicked)
    btn_go.on_click(on_go_clicked)

    # Initial display
    show_email(index[0])

    # Layout buttons and output
    navigation = HBox(
        [btn_prev, lbl_position, lbl_doc_id, btn_next, doc_id_input, btn_go]
    )
    return VBox([navigation, output])


def train_lda_model(
    corpus: list,
    dictionary: Dictionary,
    num_topics: int = 10,
    random_state: int = 100,
    update_every: int = 1,
    chunksize: int = 100,
    passes: int = 10,
    alpha: str = "auto",
) -> LdaModel:
    """
    Trains an LDA model using the provided corpus and dictionary settings. The function initializes and returns an LDA model
    from the Gensim library, configured with various parameters to control the learning process.

    Parameters
    ----------
    corpus : list
        A list of documents in the form of bag-of-words (BoW) or TF-IDF, which is used to train the LDA model. Each document
        is represented as a list of (word_id, frequency) tuples.
    dictionary : Dictionary
        Gensim dictionary object that maps word IDs to words.
    num_topics : int, optional
        Number of latent topics to be extracted from the training corpus (default is 10).
    random_state : int, optional
        Seed for random number generator to achieve reproducible results (default is 100).
    update_every : int, optional
        Number of document chunks to process before performing a model update (default is 1).
    chunksize : int, optional
        Number of documents to be used in each training chunk (default is 100).
    passes : int, optional
        Total number of training passes through the corpus (default is 10).
    alpha : str, optional
        Hyperparameter that affects the sparsity or density of the topics extracted (default is 'auto'). When set to 'auto',
        the model dynamically adjusts alpha based on the corpus.

    Returns
    -------
    LdaModel
        A trained LDA model which can be used for topic inference and other analyses.

    Examples
    --------
    >>> texts = [['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
    >>> dictionary = Dictionary(texts)
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> lda = train_lda_model(corpus, dictionary, num_topics=2, passes=15)
    >>> lda.print_topics(num_words=4)
    [(0, '0.162*"computer" + 0.162*"user" + 0.092*"response" + 0.092*"time"'),
     (1, '0.220*"computer" + 0.128*"human" + 0.128*"interface" + 0.128*"system"')]
    """
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        update_every=update_every,
        chunksize=chunksize,
        passes=passes,
        alpha=alpha,
        per_word_topics=True,
    )
    return lda_model


def get_processed_texts_and_ids(df: pd.DataFrame) -> list:
    """
    Extracts processed texts and their corresponding document IDs from a DataFrame, returning
    them as a list of tuples. This list is useful for further text processing or model training
    where document identifiers and their content need to be utilized together.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame that must contain at least the 'doc_id' and 'processed_text' columns.
        The 'processed_text' should be a string of preprocessed text for each document.

    Returns
    -------
    list of tuples
        Each tuple contains the document ID and the associated processed text split into words.
        Format: [(doc_id, [processed_text_words]), ...]

    Examples
    --------
    >>> df = pd.DataFrame({
        'doc_id': ['001', '002'],
        'processed_text': ['text data mining', 'data analysis process']
    })
    >>> processed_texts = get_processed_texts_and_ids(df)
    >>> for doc_id, text in processed_texts:
    >>>     print(f"Document ID: {doc_id}, Processed Text: {text}")
    Document ID: 001, Processed Text: ['text', 'data', 'mining']
    Document ID: 002, Processed Text: ['data', 'analysis', 'process']
    """
    return [(row["doc_id"], row["processed_text"].split()) for _, row in df.iterrows()]


def get_top_topics_per_document(
    lda_model: LdaModel, corpus: list, top_n: int = 10
) -> list:
    """
    Retrieves the top N topics for each document in the given corpus based on a trained LDA model,
    sorted by the probability of topic dominance.

    Parameters
    ----------
    lda_model : LdaModel
        The trained LDA model from which to derive the top topics.
    corpus : list of list of (int, float)
        The document corpus in the bag-of-words format; each document is represented as a list of (word_id, word_frequency) tuples.
    top_n : int, optional
        The number of top topics to retrieve for each document (default is 10).

    Returns
    -------
    list of list of (int, float)
        A list where each element is a list of tuples for a single document; each tuple consists of a topic identifier and its
        corresponding probability, indicating the dominance of the topic in that document.

    Examples
    --------
    >>> lda_model = LdaModel(corpus, num_topics=5)
    >>> corpus = [[(0, 1)], [(1, 1)]]
    >>> top_topics = get_top_topics_per_document(lda_model, corpus)
    >>> print(top_topics[0])
    [(0, 0.9), (1, 0.1)]
    """
    top_topics = []
    for doc_bow in corpus:
        sorted_topics = sorted(
            lda_model.get_document_topics(doc_bow), key=lambda x: -x[1]
        )[:top_n]
        top_topics.append(sorted_topics)
    return top_topics


def prepare_corpus_and_dictionary(texts: list) -> tuple:
    """
    Prepares a dictionary and corpus from the provided texts for use in topic modeling.

    Parameters
    ----------
    texts : list of list of str
        A list where each element is a list of tokens (words) from a single document.

    Returns
    -------
    tuple
        A tuple containing:
        - Dictionary: A Gensim Dictionary object of the unique tokens in the texts.
        - list: A list of bag-of-words (BoW) tuples for each document in the texts.

    Examples
    --------
    >>> texts = [["hello", "world"], ["hello", "gensim"]]
    >>> dictionary, corpus = prepare_corpus_and_dictionary(texts)
    >>> print(dictionary.token2id)
    >>> print(corpus)
    """
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus


def create_emails_with_topics_dataframe(
    doc_ids: list, emails_df: pd.DataFrame, top_topics_str: list
) -> pd.DataFrame:
    """
    Creates a new DataFrame from the provided document identifiers and topics, integrating email data.

    Parameters
    ----------
    doc_ids : list of str
        A list of document identifiers.
    emails_df : pd.DataFrame
        A DataFrame containing original emails, expected to have columns like 'date', 'from_line', 'subj_line', 'email_body'.
    top_topics_str : list of str
        A list of strings where each string represents the top topics identified for a document.

    Returns
    -------
    pd.DataFrame
        A DataFrame that includes the original email information along with a 'top_topics' column representing the top topics for each email.

    Examples
    --------
    >>> doc_ids = ['doc1', 'doc2']
    >>> data = {'date': ['2021-01-01', '2021-01-02'], 'from_line': ['sender@example.com', 'another@example.com'],
                'subj_line': ['Greetings', 'Update'], 'email_body': ['Hello world', 'Latest news']}
    >>> emails_df = pd.DataFrame(data)
    >>> top_topics_str = ['Topic 1: 0.70; Topic 3: 0.30', 'Topic 2: 0.65; Topic 5: 0.35']
    >>> emails_with_topics_df = create_emails_with_topics_dataframe(doc_ids, emails_df, top_topics_str)
    >>> print(emails_with_topics_df)
    """
    df_emails_with_topics = pd.DataFrame(
        {
            "doc_id": doc_ids,
            "date": emails_df["date"],
            "from_line": emails_df["from_line"],
            "subj_line": emails_df["subj_line"],
            "email_body": emails_df["email_body"],
            "top_topics": top_topics_str,
        }
    )
    return df_emails_with_topics


def train_lda_model(
    corpus: list,
    dictionary: Dictionary,
    num_topics: int = 10,
    random_state: int = 100,
    chunksize: int = 2000,
    passes: int = 10,
    alpha: str = "symmetric",
    workers: int = None,
) -> LdaMulticore:
    """
    Trains an LDA (Latent Dirichlet Allocation) model using multiple cores and displays training progress.

    Parameters
    ----------
    corpus : list
        The corpus to train the LDA model on, typically as a list of bag-of-words (word_id, word_frequency) tuples.
    dictionary : Dictionary
        Gensim Dictionary object mapping IDs to words, used for mapping word IDs to words in the corpus.
    num_topics : int, optional
        The number of latent topics to be extracted from the training corpus, default is 10.
    random_state : int, optional
        Seed for random number generator to achieve results reproducibility, default is 100.
    chunksize : int, optional
        Number of documents to be used in each training chunk, default is 2000.
    passes : int, optional
        Number of passes through the corpus during training, default is 10.
    alpha : str, optional
        Hyperparameter affecting the sparsity/thickness of the topics. Default is 'symmetric'.
    workers : int, optional
        The number of worker processes to be used for parallelization. If None, all available cores will be used.

    Returns
    -------
    LdaMulticore
        The trained LDA model.

    Examples
    --------
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import LdaMulticore
    >>> dictionary = Dictionary(documents)
    >>> corpus = [dictionary.doc2bow(text) for text in documents]
    >>> lda_model = train_lda_model(corpus, dictionary, num_topics=5, passes=15, workers=4)

    Notes
    -----
    This function initializes the LDA model with one pass to ensure the model's proper setup and then manually
    updates the model in a loop for the remaining number of passes, showing a tqdm progress bar for visibility.
    """
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=chunksize,
        passes=1,  # Initialize model with a single pass.
        alpha=alpha,
        workers=workers,
    )

    # tqdm progress bar for the remaining passes, if more than one pass is required.
    if passes > 1:
        for pass_idx in tqdm(range(1, passes), desc="Training LDA Model"):
            lda_model.update(corpus)  # Update the model in subsequent passes.

    return lda_model


def display_topic_words(lda_model: LdaModel, num_words: int = 10) -> pd.DataFrame:
    """
    Extracts the top words for each topic from an LDA model and formats them into a DataFrame.

    Parameters
    ----------
    lda_model : LdaModel
        The trained LDA model from which to extract the topic words.
    num_words : int, optional
        The number of top words to retrieve for each topic, default is 10.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing two columns: 'Category' which lists the topic IDs and 'Key Terms'
        which lists the top words associated with each topic.

    Examples
    --------
    >>> from gensim.models import LdaModel
    >>> lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    >>> topics_df = display_topic_words(lda_model, num_words=10)
    >>> print(topics_df)

    Notes
    -----
    The function iterates through each topic generated by the LDA model, retrieving the top words
    according to their probability of association with the topic. These words are then formatted into
    a string and combined with the topic ID in a pandas DataFrame for easy viewing or further analysis.
    """

    def extract_topic_string(words):
        """Helper function to convert topic words and probabilities into a formatted string."""
        topic_str = ", ".join([word for word, _ in words])

        topic_str = ""
        for topic, prob in words:
            topic_str += f" ,{topic}"
        topic_str = topic_str.lstrip(" ,")
        return topic_str

    topic_id_list = []
    topic_string_list = []

    for topic_id in range(lda_model.num_topics):

        words = lda_model.show_topic(topicid=topic_id, topn=num_words)
        topic_str = extract_topic_string(words)

        topic_id_list.append(topic_id)
        topic_string_list.append(topic_str)

    lda_terms_df = pd.DataFrame(
        {"Category": topic_id_list, "Key Terms": topic_string_list}
    )
    return lda_terms_df


def train_models_and_find_optimal(
    dictionary: Dictionary,
    corpus: List[List[tuple]],
    texts: List[List[str]],
    start: int = 2,
    limit: int = 22,
    step: int = 4,
    workers: int = 4,
) -> Tuple[LdaMulticore, int, float]:
    """
    Trains multiple LDA models with varying numbers of topics to find the optimal model based on coherence scores.

    Parameters
    ----------
    dictionary : Dictionary
        Gensim dictionary object of the corpus.
    corpus : List[List[tuple]]
        List of documents represented as bag-of-words.
    texts : List[List[str]]
        Tokenized texts used for coherence score calculation.
    start : int, optional
        Starting number of topics, by default 2.
    limit : int, optional
        The maximum number of topics to test, by default 22.
    step : int, optional
        Step size to iterate through the number of topics, by default 4.
    workers : int, optional
        Number of worker processes to train the LDA models, by default 4.

    Returns
    -------
    Tuple[LdaMulticore, int, float]
        A tuple containing the best LDA model, the optimal number of topics,
        and the highest coherence score achieved.

    Examples
    --------
    >>> dictionary = Dictionary(texts)
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> lda_model, num_topics, coherence = train_models_and_find_optimal(
            dictionary, corpus, texts, start=2, limit=20, step=2, workers=4)
    >>> print("Best model has", num_topics, "topics with coherence score of", coherence)

    Notes
    -----
    This function iterates through different numbers of topics, training an LDA model for each configuration
    and calculating its coherence. The highest coherence score determines the best model, which is returned
    along with its number of topics and coherence value. A plot is also displayed to visually inspect coherence
    trends across different topic counts.
    """
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step), desc="Training LDA Models"):
        model = LdaMulticore(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=100,
            chunksize=2000,
            passes=10,
            alpha="asymmetric",
            workers=workers,
        )
        model_list.append((num_topics, model))
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append((num_topics, coherencemodel.get_coherence()))

    # Find the model with the highest coherence
    best_num_topics, best_coherence = max(coherence_values, key=lambda x: x[1])
    best_model = [
        model for num_topics, model in model_list if num_topics == best_num_topics
    ][0]

    # Plot coherence scores
    x = [num_topics for num_topics, _ in coherence_values]
    y = [coherence for _, coherence in coherence_values]
    plt.plot(x, y)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.scatter(best_num_topics, best_coherence, color="red")  # Mark the best model
    plt.legend(["Coherence Values", "Best Model"], loc="best")
    plt.title("Coherence Scores by Number of Topics")
    plt.show()

    return best_model, best_num_topics, best_coherence


def perform_lda_topic_modelling(
    emails_df: pd.DataFrame,
    processed_text_col: str = "processed_text",
    num_topics: int = 10,
    random_state: int = 100,
    chunksize: int = 2000,
    passes: int = 10,
    alpha: str = "symmetric",
    workers: int = 4,
) -> Tuple[pd.DataFrame, LdaModel]:
    """
    Performs Latent Dirichlet Allocation (LDA) topic modeling on a collection of emails to identify prevalent topics
    within the corpus. This function enriches the input DataFrame by appending a column that lists the most significant
    topics in each document, based on their contribution weights.

    Parameters:
    ----------
    emails_df : pandas.DataFrame
        DataFrame containing the emails, with at least one column of preprocessed text.
    processed_text_col : str, default 'processed_text'
        Name of the column in `emails_df` that contains preprocessed text for topic modeling.
    num_topics : int, default 10
        The number of distinct topics to identify in the LDA model.
    random_state : int, default 100
        Seed for the random number generator for reproducibility.
    chunksize : int, default 2000
        Number of documents to consider at once in the training algorithm.
    passes : int, default 10
        Number of training passes through the corpus.
    alpha : {'symmetric', 'asymmetric', 'auto'} or list, default 'symmetric'
        Hyperparameter affecting document-topic density. Can be specified as a list for asymmetric priors.
    workers : int, default 4
        Number of worker processes to use for parallelization. If None, uses all available cores minus one.

    Returns:
    -------
    tuple
        A tuple containing:
        - DataFrame: The input DataFrame enriched with a new column 'top_topics' that lists the dominant topics and their
          weights for each document.
        - model: The trained LdaModel, which can be used for further analysis or visualization of topics.

    Examples:
    --------
    >>> emails_df = pd.DataFrame({
            'processed_text': ["text about health", "text about finance"]
        })
    >>> enriched_df, lda_model = perform_lda_topic_modelling(emails_df)
    >>> print(enriched_df['top_topics'].head())

    Notes:
    -----
    This function is crucial for understanding large collections of text by breaking them down into manageable themes
    or topics. It's particularly useful in exploratory data analysis and natural language processing applications where
    the themes of documents need to be understood quickly.
    """
    # Prepare texts and document IDs
    texts = [doc.split() for doc in emails_df[processed_text_col]]

    # Create dictionary and corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Optionally apply TF-IDF transformation
    corpus_tfidf = TfidfModel(corpus)[corpus]

    # Train the LDA model
    lda_model_tfidf = train_lda_model(
        corpus_tfidf,
        dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=chunksize,
        passes=passes,
        alpha=alpha,
        workers=workers,
    )

    # Extract top topics for each document
    top_topics_per_document = [
        lda_model_tfidf.get_document_topics(item) for item in corpus_tfidf
    ]
    top_topics_str = [
        "; ".join([f"Topic {topic_num}: {prob:.2f}" for topic_num, prob in doc])
        for doc in top_topics_per_document
    ]

    # Enrich DataFrame with top topics
    enriched_emails_df = emails_df.copy()
    enriched_emails_df["top_topics"] = top_topics_str

    return enriched_emails_df, lda_model_tfidf, corpus, dictionary, corpus_tfidf


def get_top_topics_str_for_each_document(
    lda_model: LdaModel, corpus: List[List[tuple]], top_n: int = 10
) -> List[str]:
    """
    Generates a string representation of the top topics for each document in a corpus, as determined by an LDA model.
    Each topic is listed along with its probability in descending order, formatted for easy readability.

    Parameters:
    ----------
    lda_model : gensim.models.LdaModel
        The trained LDA model used to analyze the documents.
    corpus : iterable of iterable of (int, int)
        The document corpus in bag-of-words format; each document is represented as a list of (word_id, word_frequency) tuples.
    top_n : int, optional
        The number of top topics to include for each document (default is 10).

    Returns:
    -------
    list of str
        A list where each element is a string representing the top topics for a corresponding document in the corpus.
        Each string is formatted as "Topic X: Y.YY; Topic Z: W.WW", where X and Z are topic numbers and Y.YY, W.WW are
        the associated probabilities.

    Example:
    --------
    >>> lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
    >>> corpus = [[(0, 1), (1, 1)], [(2, 1), (3, 1)]]
    >>> top_topics_str = get_top_topics_str_for_each_document(lda_model, corpus)
    >>> print(top_topics_str[0])
    'Topic 1: 0.70; Topic 3: 0.30'
    """
    top_topics_per_document = get_top_topics_per_document(
        lda_model, corpus, top_n=top_n
    )
    top_topics_str = [
        "; ".join([f"Topic {topic_num}: {prob:.2f}" for topic_num, prob in topics])
        for topics in top_topics_per_document
    ]
    return top_topics_str


def train_doc2vec_model(
    tokenized_docs: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 40,
    workers: int = 4,
) -> Doc2Vec:
    """
    Trains a Doc2Vec model on a given corpus of tokenized documents. Doc2Vec is an unsupervised algorithm to generate
    vector representations for documents, which can be used for various applications such as similarity search or
    document clustering.

    Parameters:
    ----------
    tokenized_docs : List[List[str]]
        A list of tokenized documents, where each document is represented as a list of words.
    vector_size : int, optional
        The dimensionality of the document vectors (default is 100).
    window : int, optional
        The maximum distance between the current and predicted word within a sentence (default is 5).
    min_count : int, optional
        Ignores all words with total frequency lower than this (default is 5).
    epochs : int, optional
        Number of iterations (epochs) over the corpus (default is 40).
    workers : int, optional
        The number of worker threads to train the model, which speeds up training on multicore machines (default is 4).

    Returns:
    -------
    gensim.models.doc2vec.Doc2Vec
        The trained Doc2Vec model.

    Example:
    --------
    >>> tokenized_docs = [['hello', 'world'], ['example', 'text']]
    >>> model = train_doc2vec_model(tokenized_docs)
    """
    tagged_data = [
        TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_docs)
    ]
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )
    model.build_vocab(tagged_data)

    # Train model
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def infer_doc2vec_vector(
    model: Doc2Vec, tokenized_doc: List[str], epochs: int = 40
) -> np.ndarray:
    """
    Infers a vector for a single document using a pre-trained Doc2Vec model, which is useful for obtaining document
    representations for new or unseen data.

    Parameters:
    ----------
    model : gensim.models.doc2vec.Doc2Vec
        The trained Doc2Vec model from which to infer the document vector.
    tokenized_doc : List[str]
        The tokenized document (a list of words) for which the vector will be inferred.
    epochs : int, optional
        The number of epochs to use for inference (default is 40).

    Returns:
    -------
    numpy.ndarray
        The inferred document vector.

    Example:
    --------
    >>> model = train_doc2vec_model([['hello', 'world'], ['example', 'text']])
    >>> vector = infer_doc2vec_vector(model, ['hello', 'example'])
    """
    return model.infer_vector(tokenized_doc, epochs=epochs)


def categorize_documents(
    df: pd.DataFrame,
    processed_text_col: str = "processed_text",
    vector_size: int = 100,
    min_count: int = 5,
    epochs: int = 40,
    n_clusters: int = 5,
) -> pd.DataFrame:
    """
    Categorizes documents into clusters based on the semantic content of their processed text using Doc2Vec embeddings
    followed by KMeans clustering. This function aims to group documents that share similar themes and extracts key terms
    that describe these clusters, facilitating an easier understanding of the thematic structure within a large text corpus.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the documents to be categorized.
    processed_text_col : str, optional
        The name of the column in 'df' that contains the preprocessed text.
    vector_size : int, optional
        The dimensionality of the Doc2Vec embeddings.
    min_count : int, optional
        The minimum count of words required for them to be considered by the Doc2Vec model.
    epochs : int, optional
        The number of training epochs for the Doc2Vec model.
    n_clusters : int, optional
        The number of clusters to form with KMeans clustering.

    Returns:
    -------
    pd.DataFrame
        A DataFrame that includes the original documents along with their assigned cluster category and key terms
        representative of each category.

    Steps:
    1. Train a Doc2Vec model using the preprocessed text to create vector representations for each document.
    2. Apply KMeans clustering to these vectors to form specified number of clusters.
    3. Identify documents closest to each cluster centroid using cosine similarity.
    4. Extract frequent and unique key terms from top documents within each cluster to serve as descriptors.
    5. Append the cluster labels and key terms to the original DataFrame and return this enhanced DataFrame.

    Note:
    -----
    This method effectively turns the unstructured text data into structured form, enabling further analytical and
    machine learning applications. The function depends on the Gensim library for Doc2Vec and scikit-learn for KMeans.
    """

    # Train Doc2Vec model
    tokenized_docs = [doc.split() for doc in df[processed_text_col]]
    d2v_model = train_doc2vec_model(
        tokenized_docs,
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        workers=4,
    )

    # Infer vectors for all documents
    vectors = np.array([d2v_model.infer_vector(doc) for doc in tokenized_docs])

    # Cluster document vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectors)
    df["category"] = kmeans.labels_

    # Calculate cluster centroids
    centroids = kmeans.cluster_centers_

    # Extract top terms for each cluster
    terms = {i: [] for i in range(n_clusters)}
    for i, centroid in enumerate(centroids):
        cos_similarities = cosine_similarity(centroid.reshape(1, -1), vectors)
        top_docs_indices = cos_similarities.argsort()[0][
            -3:
        ]  # Top 3 docs for this cluster
        for idx in top_docs_indices:
            terms[i].extend(tokenized_docs[idx])

    # Identify unique, frequent terms for each cluster
    for cluster, tokens in terms.items():
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        sorted_indices = counts.argsort()[::-1][:5]  # Top 5 terms
        terms[cluster] = unique_tokens[sorted_indices]

    # Create final DataFrame
    final_df = df.copy()
    final_df["key_terms"] = final_df["category"].apply(lambda x: ", ".join(terms[x]))

    return final_df


def display_categories_and_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Displays a summary of categories and their associated key terms from a DataFrame containing
    categorized document information. This function assumes each category in the DataFrame has consistent
    key terms across different entries.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame that must include at least two columns: 'category' and 'key_terms'.
        - 'category': A column containing category identifiers.
        - 'key_terms': A column containing a string or list of key terms associated with each category.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame where each row corresponds to a unique category and its associated key terms.

    Examples:
    --------
    >>> df = pd.DataFrame({
        'category': ['Technology', 'Technology', 'Health', 'Health'],
        'key_terms': ['AI, Machine Learning', 'AI, Machine Learning', 'Disease Prevention', 'Disease Prevention']
    })
    >>> categories_df = display_categories_and_terms(df)
    >>> print(categories_df)

    Notes:
    -----
    This function is particularly useful for quickly summarizing the categorical breakdown of document datasets,
    especially when each category has been assigned specific key terms. The function simplifies the visualization
    of these relationships in a tabular form.
    """
    # Get unique categories
    unique_categories = df["category"].unique()

    # Create a display DataFrame
    categories_df = pd.DataFrame(columns=["Category", "Key Terms"])

    for category in unique_categories:
        # Get the key terms for the current category
        key_terms = df[df["category"] == category]["key_terms"].iloc[
            0
        ]  # Assuming key terms are consistent within each category

        # Append to the display DataFrame
        categories_df = categories_df.append(
            {"Category": category, "Key Terms": key_terms}, ignore_index=True
        )

    # Display the DataFrame
    return categories_df


def interactive_email_and_terms_viewer(df_emails: pd.DataFrame, data_path: str) -> VBox:
    """
    Creates an interactive viewer in a Jupyter Notebook to display emails alongside their associated key terms.
    This viewer allows navigation through a list of emails stored as .txt files in a specified directory,
    which are indexed and linked to their metadata stored in a DataFrame. It also allows direct navigation to any
    email by entering its document ID.

    Parameters:
    ----------
    df_emails : pd.DataFrame
        A DataFrame containing the emails' metadata. It must include at least two columns: 'doc_id' and 'key_terms',
        where 'doc_id' corresponds to the filename of the email file (minus the .txt extension) and 'key_terms'
        contains the associated key terms or categories.
    data_path : str
        The path to the directory where the email .txt files are stored. Each file should be named with a 'doc_id' from
        the DataFrame and a .txt extension.

    Returns:
    -------
    VBox
        A VBox widget that contains navigation controls and displays the content of the emails and their associated
        key terms. This widget can be displayed in a Jupyter Notebook environment.

    Examples:
    --------
    >>> df_emails = pd.DataFrame({
        'doc_id': ['email1', 'email2'],
        'key_terms': ['term1, term2', 'term3, term4']
    })
    >>> data_path = '/path/to/emails'
    >>> email_viewer = interactive_email_and_terms_viewer(df_emails, data_path)
    >>> display(email_viewer)

    Notes:
    -----
    The viewer includes 'Previous' and 'Next' buttons to navigate through the emails, a 'Go' button to jump to a
    specific email, and a display area that shows the content of the current email and its key terms side by side.
    It is ideal for exploring datasets of emails where understanding the context and content is crucial.
    """
    files = df_emails["doc_id"].apply(lambda x: f"{x}.txt").tolist()

    index = [0]  # Mutable object to keep track of the index in a closure

    # Widgets
    output = Output(layout={"border": "1px solid black", "width": "100%"})
    btn_prev = Button(
        description="Previous", layout=Layout(width="100px", height="30px")
    )
    btn_next = Button(description="Next", layout=Layout(width="100px", height="30px"))
    doc_id_input = Text(
        description="Go to ID:",
        placeholder="Enter Document ID",
        layout=Layout(width="200px"),
    )

    btn_go = Button(description="Go", layout=Layout(width="80px", height="30px"))
    lbl_position = Label()
    lbl_doc_id = Label()

    def update_labels():
        lbl_position.value = f"Document {index[0] + 1} of {len(files)}"
        lbl_doc_id.value = (
            f"Doc ID: {files[index[0]][:-4]}"  # Remove '.txt' extension for display
        )

    def show_email(idx):
        output.clear_output()
        file_path = os.path.join(data_path, files[idx])
        doc_id = files[idx][:-4]
        email_row = df_emails[df_emails["doc_id"] == doc_id].iloc[0]
        original_text = view_file(file_path)
        key_terms = email_row["key_terms"]

        with output:
            display(
                HTML(
                    f"""
            <style>
                .email-view {{ width: 49%; overflow-wrap: break-word; white-space: pre-wrap; }}
                table {{ width: 100%; table-layout: fixed; }}
                td {{ vertical-align: top; }}
            </style>
            <table>
                <tr>
                    <td class="email-view"><b>Original Email:</b><br>{original_text}</td>
                    <td class="email-view"><b>Key Terms for Category:</b><br>{key_terms}</td>
                </tr>
            </table>
            """
                )
            )
        update_labels()

    btn_prev.on_click(lambda b: navigate(-1))
    btn_next.on_click(lambda b: navigate(1))
    btn_go.on_click(go_to_doc)

    def navigate(direction):

        new_index = max(0, min(len(files) - 1, index[0] + direction))
        if new_index != index[0]:
            index[0] = new_index
            show_email(index[0])

    def go_to_doc(b):
        try:
            target_id = doc_id_input.value.strip() + ".txt"
            target_index = files.index(target_id)
            index[0] = target_index
            show_email(index[0])

        except ValueError:
            output.clear_output()
            with output:
                print("Document ID not found!")

    show_email(index[0])
    navigation = HBox(
        [btn_prev, lbl_position, lbl_doc_id, btn_next, doc_id_input, btn_go]
    )
    return VBox([navigation, output])
