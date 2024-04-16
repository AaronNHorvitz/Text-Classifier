"""
Utilities Module for TopicMiner

This module provides utility functions for processing email data for the TopicMiner application.
It includes functions to convert date formats, generate formatted email text with proper word wrapping,
and save the formatted emails as text files.

Functions:
- convert_date_format(date_str): Converts a date string from 'YYYY-MM-DD' format to 'Month DD, YYYY' format.
- generate_email_text(email_date, email_recipient, email_from, email_subject, email_body): 
  Generates a formatted email text with a word-wrapped body, including headers such as To, From, Sent, and Subject.
- format_and_save_emails(df, output_dir): Processes a DataFrame containing email data, formats each email,
  and saves it as a text file in the specified directory.

The module uses Python's built-in datetime and textwrap libraries to handle date conversions and text formatting,
ensuring that the emails are easily readable and maintain a standard layout.

Example Usage:
To use these utilities, import the module and call functions like `convert_date_format()` with a date string,
or `format_and_save_emails()` with a DataFrame of emails and an output directory path.

Dependencies:
- pandas: Used for handling data in DataFrame format.
- numpy: Utilized for handling NaN values effectively.
- datetime: Needed for date manipulations.
- textwrap: Employed to ensure that the email body is wrapped at word boundaries and does not exceed 100 characters per line.

"""

# Standard library imports
import json
import os
import re
import textwrap

# Third-party library imports
import chardet
import numpy as np
import pandas as pd

# Date and time utilities
from datetime import datetime

# Natural Language Processing tools from NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Utilities for display and progress tracking
from IPython.display import display, HTML, clear_output
from ipywidgets import Button, HBox, VBox, Output, Layout, Text, Label
from tqdm import tqdm


def convert_date_format(date_str: str) -> str:
    """
    Convert a date string from 'YYYY-MM-DD' format to 'Month DD, YYYY' format.

    This function takes a date string in the ISO 8601 date format (YYYY-MM-DD) and converts
    it to a more readable string format that includes the full month name, the day, and the year.

    Parameters
    ----------
    date_str : str
        A string representing the date in 'YYYY-MM-DD' format. This should be a valid date string
        that conforms to the ISO 8601 date format.

    Returns
    -------
    str
        A string representing the date in 'Month DD, YYYY' format. If `date_str` is NaN or an invalid
        date, the function will return NaN.

    Examples
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


def generate_email_text(
    email_date: str,
    email_recipient: str,
    email_from: str,
    email_subject: str,
    email_body: str,
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
    email_body = textwrap.fill(email_body, width=100)

    # Construct the email string
    email_str = f"""
To: {email_recipient}
From: {email_from}
Sent: {email_date}
Subject: {email_subject}

{email_body}
    """
    return email_str.strip()


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


def read_text_file(file_path: str) -> list:
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


def parse_email_text_file_components(text_file_contents: list) -> tuple:
    """
    Parse the components of an email from a list of lines of the email text.

    Parameters
    ----------
    text_file_contents : list of str
        Lines from a text file containing an email. Each line should correspond to one line of the text file.

    Returns
    -------
    tuple
        Contains 'To', 'From', 'Sent date', 'Subject', and 'Body' of the email as strings.
        Each component is extracted based on its prefix and the body is collected after an empty line.

    Examples
    --------
    >>> email_lines = [
        "To: Example Recipient",
        "From: Example Sender",
        "Sent: January 01, 2020",
        "Subject: Example Subject",
        "",
        "This is the first line of the email body.",
        "This is the second line of the email body."
    ]
    >>> recipient, sender, date, subject, body = parse_email_text_file_components(email_lines)
    >>> print(recipient)
    'Example Recipient'
    >>> print(body)
    'This is the first line of the email body.\nThis is the second line of the email body.'

    Notes
    -----
    This function assumes a specific format of the email text, where headers are provided
    with 'To:', 'From:', 'Sent:', and 'Subject:' prefixes, followed by an empty line before
    the body of the email starts.
    """
    # Initialize default values for email components
    email_recipient = "Unknown Recipient"
    email_from = "Unknown Sender"
    email_date = "Unknown Date"
    email_subject = "No Subject"
    email_body = []

    # Flag to indicate start of body
    body_start_found = False

    for line in text_file_contents:
        stripped_line = line.strip()
        if stripped_line.startswith("To:"):
            email_recipient = stripped_line.replace("To:", "").strip()
        elif stripped_line.startswith("From:"):
            email_from = stripped_line.replace("From:", "").strip()
        elif stripped_line.startswith("Sent:"):
            email_date = stripped_line.replace("Sent:", "").strip()
        elif stripped_line.startswith("Subject:"):
            email_subject = stripped_line.replace("Subject:", "").strip()
        elif stripped_line == "":
            body_start_found = True  # Empty line indicates the start of the body
        elif body_start_found:
            email_body.append(stripped_line)  # Append line to body

    email_body = "\n".join(email_body)  # Join all body lines into a single string

    return email_recipient, email_from, email_date, email_subject, email_body


def add_unwanted_text(file_path: str, new_text: str) -> None:
    """
    Adds a new unwanted text string to a JSON file. If the file does not exist, this function
    creates a new one with the given text. It also ensures that duplicates are not added.

    Parameters
    ----------
    file_path : str
        Path to the JSON file storing unwanted texts.
    new_text : str
        New text string to add to the file.

    Returns
    -------
    None

    Examples
    --------
    >>> add_unwanted_text("path/to/unwanted_texts.json", "Example unwanted text")
    Text added successfully.

    Notes
    -----
    If the JSON file does not exist at the specified `file_path`, the function will create
    the file and initialize it with the `new_text` as the first entry. If the JSON file
    exists but is corrupted or empty, the function initializes a new list. This function
    ensures that the directory for the specified path exists before attempting to write
    to the file.
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


def delete_unwanted_text(file_path: str, text_to_delete: str) -> None:
    """
    Deletes an unwanted text string from a JSON file. If the file does not exist or is empty,
    it notifies the user. It also handles the removal operation safely by checking the presence
    of the text to delete.

    Parameters
    ----------
    file_path : str
        Path to the JSON file storing unwanted texts.
    text_to_delete : str
        Text string to delete from the file.

    Returns
    -------
    None

    Examples
    --------
    >>> delete_unwanted_text("path/to/unwanted_texts.json", "Example unwanted text")
    Text removed successfully.

    Notes
    -----
    The function first checks if the JSON file exists. If not, it notifies the user and returns without
    modifying anything. If the file exists but contains errors (e.g., it's corrupted), the user is informed
    of the issue. If the specified text to delete is found in the list of unwanted texts, it is removed
    and the updated list is saved back to the JSON file. If the text is not found, a notification is printed.
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


def load_unwanted_text(file_path: str) -> list:
    """
    Reads all unwanted text strings from a JSON file and returns them as a list. If the file does not exist or is empty,
    it notifies the user and returns an empty list.

    Parameters
    ----------
    file_path : str
        Path to the JSON file storing unwanted texts.

    Returns
    -------
    list
        A list of unwanted text strings. Returns an empty list if the file is not found, is empty, or corrupted.

    Examples
    --------
    >>> load_unwanted_texts("path/to/unwanted_texts.json")
    ['Example unwanted text', 'Another unwanted phrase']

    Notes
    -----
    The function checks if the JSON file exists at the specified path. If not, it returns an empty list and notifies
    the user of the missing file. If the file is found but cannot be read properly due to corruption or being empty,
    it also returns an empty list and notifies the user. This function is typically used to load unwanted texts for
    preprocessing text data, where these phrases are to be removed.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print("JSON file does not exist.")
        return []

    try:
        with open(file_path, "r") as file:
            unwanted_texts = json.load(file)
            return unwanted_texts
    except json.JSONDecodeError:
        print("JSON file containing unwanted texts is empty or corrupted.")
        return []

def clean_email_body(email_body: str, unwanted_texts_file_path: str) -> str:
    """
    Cleans the email body by removing all unwanted text phrases defined in a JSON file.
    
    Parameters:
    ----------
    email_body : str
        The original body of the email which may contain unwanted phrases.
    unwanted_texts_file_path : str
        Path to the JSON file that contains a list of unwanted text strings.
    
    Returns:
    -------
    str
        The cleaned email body with all unwanted texts removed.
    """
    # Load the list of unwanted texts from the specified JSON file
    unwanted_texts = load_unwanted_text(unwanted_texts_file_path)
    
    # Remove each unwanted phrase from the email body
    cleaned_body = email_body
    for unwanted_text in unwanted_texts:
        cleaned_body = cleaned_body.replace(unwanted_text, "")
    
    return cleaned_body

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
    # Remove unwanted phrases from text
    text = clean_email_body(text, unwanted_texts_file_path)
    
    # Convert text to lowercase to standardize it
    text = text.lower()
    
    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Load English stopwords
    stop_words = set(stopwords.words("english"))
    
    # Filter out stopwords, non-alphabetic tokens, and single-character tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]
    
    # Initialize the NLTK lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join words back into a single string
    preprocessed_text = " ".join(tokens)
    
    return preprocessed_text


def email_viewer_widget(processed_files_directory):
    """
    """
    files = sorted([f for f in os.listdir(processed_files_directory) if f.endswith('.txt')])
    if not files:
        print("No text files found in the directory.")
        return
    
    output_area = widgets.Output(layout={'border': '1px solid black', 'width': '100%', 'height': '300px'})
    doc_id_input = widgets.Text(placeholder='Enter Document ID', description='Go to ID:')
    go_button = widgets.Button(description='Go')
    prev_button = widgets.Button(description='Previous')
    next_button = widgets.Button(description='Next')
    navigation_info = widgets.Label(value=f"Document 1 of {len(files)}")
    
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def update_display(index):
        file_path = os.path.join(processed_files_directory, files[index])
        email_content = read_file(file_path)
        with output_area:
            clear_output(wait=True)
            display(email_content)
        navigation_info.value = f"Document {index + 1} of {len(files)}"
    
    def on_prev_button_clicked(b):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            update_display(current_index)
    
    def on_next_button_clicked(b):
        nonlocal current_index
        if current_index < len(files) - 1:
            current_index += 1
            update_display(current_index)
    
    def on_go_button_clicked(b):
        try:
            target_id = f"{doc_id_input.value}.txt"
            target_index = files.index(target_id)
            update_display(target_index)
            current_index = target_index
        except ValueError:
            with output_area:
                clear_output(wait=True)
                print("Document ID not found!")
    
    prev_button.on_click(on_prev_button_clicked)
    next_button.on_click(on_next_button_clicked)
    go_button.on_click(on_go_button_clicked)
    
    current_index = 0
    update_display(current_index)
    
    navigation_buttons = widgets.HBox([prev_button, navigation_info, next_button])
    search_box = widgets.HBox([doc_id_input, go_button])
    interface = widgets.VBox([search_box, navigation_buttons, output_area])
    
    return interface


def email_viewer(doc_id, data_path, unwanted_text_file_path):
    """
    Displays an email text comparison in a formatted table including original, cleaned, and preprocessed versions.

    Parameters:
    - doc_id (str): Document ID for the email.
    - data_path (str): Path to the directory containing the email files.
    - unwanted_text_file_path (str): Path to the JSON file containing unwanted texts.

    This function reads the email from a file, parses it, cleans it of unwanted texts, preprocesses it for modeling,
    and then displays all versions side by side for comparison.
    """
    # Construct full path to the email file
    text_file_path = os.path.join(data_path, f"{doc_id}.txt")

    try:
        # Retrieve and parse the original email text
        text_file_contents = read_text_file(text_file_path)
        email_recipient, email_from, email_date, email_subject, email_body = parse_email_text_file_components(text_file_contents)
        reconstructed_email = f"To: {email_recipient}\nFrom: {email_from}\nSent: {email_date}\n\nSubject: {email_subject}\n\n{email_body}"
        reconstructed_email = ''.join(reconstructed_email)

        # Load unwanted texts for highlighting
        unwanted_texts = load_unwanted_text(unwanted_text_file_path)
        
        # Highlight unwanted text in orange
        highlighted_email_body = reconstructed_email
        for phrase in unwanted_texts:
            highlighted_email_body = highlighted_email_body.replace(phrase, f"<mark style='background-color: orange;'>{phrase}</mark>")

        # Remove unwanted text to prepare a cleaned email. 
        cleaned_email = clean_email_body(reconstructed_email, unwanted_text_file_path)
        cleaned_email = ''.join(cleaned_email)

        cleaned_email_body = clean_email_body(email_body, unwanted_text_file_path)
        normalized_text = preprocess_text(cleaned_email_body, unwanted_text_file_path)
        normalized_text = ''.join(normalized_text)

        # Create HTML content for display
        comparison_html = f"""
        <style>
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
            th {{background-color: #f2f2f2;}}
            pre {{white-space: pre-wrap; word-wrap: break-word; text-align: left; margin: 0;}}
            mark {{background-color: orange;}}
            .text-container {{text-align: left; font-size: 12px;}}  /* Smaller font size */
        </style>
        <table>
            <tr>
                <th>Original Email (Highlighted Unwanted Text)</th>
                <th>Removed Unwanted Text</th>
                <th>Normalized Text</th>
            </tr>
            <tr>
                <td><div class="text-container"><pre>{highlighted_email_body}</pre></div></td>
                <td><div class="text-container"><pre>{cleaned_email}</pre></div></td>
                <td><div class="text-container"><pre>{normalized_text}</pre></div></td>
            </tr>
        </table>
        """

        # Display the comparison in HTML format
        return comparison_html

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# # Utilities for display and progress tracking
# from IPython.display import display, HTML, clear_output
# from ipywidgets import Button, HBox, VBox, Output, Layout, Text, Label

# from IPython.display import display, HTML, clear_output
# import ipywidgets as widgets
# import os

# def email_viewer_widget(processed_files_directory, unwanted_text_file_path):
#     files = sorted([f for f in os.listdir(processed_files_directory) if f.endswith('.txt')])
#     if not files:
#         print("No text files found in the directory.")
#         return

#     output_area = widgets.Output(layout={'border': '1px solid black', 'width': '100%', 'height': '500px'})
#     doc_id_input = widgets.Text(placeholder='Enter Document ID', description='Go to ID:')
#     go_button = widgets.Button(description='Go')
#     prev_button = widgets.Button(description='Previous')
#     next_button = widgets.Button(description='Next')
#     navigation_info = widgets.Label(value=f"Document 1 of {len(files)}")
#     current_index = [0]  # Use a list to hold current index to modify inside nested functions

#     def update_display():
#         doc_id = files[current_index[0]][:-4]  # Remove '.txt' extension
#         comparison_html = email_viewer(doc_id, processed_files_directory, unwanted_text_file_path)
#         with output_area:
#             clear_output(wait=True)  # Clear previous outputs
#             display(HTML(comparison_html))  # Display new HTML content within the widget
#         navigation_info.value = f"Document {current_index[0] + 1} of {len(files)}"

#     def on_prev_button_clicked(b):
#         if current_index[0] > 0:
#             current_index[0] -= 1
#             update_display()

#     def on_next_button_clicked(b):
#         if current_index[0] < len(files) - 1:
#             current_index[0] += 1
#             update_display()

#     def on_go_button_clicked(b):
#         try:
#             target_id = f"{doc_id_input.value}.txt"
#             target_index = files.index(target_id)
#             current_index[0] = target_index
#             update_display()
#         except ValueError:
#             with output_area:
#                 clear_output(wait=True)
#                 print("Document ID not found!")

#     prev_button.on_click(on_prev_button_clicked)
#     next_button.on_click(on_next_button_clicked)
#     go_button.on_click(on_go_button_clicked)

#     update_display()  # Initial display update

#     navigation_buttons = widgets.HBox([prev_button, navigation_info, next_button])
#     search_box = widgets.HBox([doc_id_input, go_button])
#     interface = widgets.VBox([search_box, navigation_buttons, output_area])

#     return interface

# from IPython.display import display, HTML, clear_output
# import ipywidgets as widgets
# import os

def email_viewer_widget(processed_files_directory, unwanted_text_file_path):
    files = sorted([f for f in os.listdir(processed_files_directory) if f.endswith('.txt')])
    if not files:
        print("No text files found in the directory.")
        return

    output_area = Output(layout={'border': '1px solid black', 'width': '100%', 'height': '500px'})
    doc_id_input = Text(placeholder='Enter Document ID', description='Go to ID:')
    go_button = Button(description='Go')
    prev_button = Button(description='Previous')
    next_button = Button(description='Next')
    add_text_input = Text(placeholder='Enter text to add/remove', description='Manage Text:')
    add_button = Button(description='Add Unwanted Text')
    delete_button = Button(description='Delete Unwanted Text')
    refresh_button = Button(description='Refresh View')
    navigation_info = Label(value=f"Document 1 of {len(files)}")
    current_index = [0]  # Use a list to hold current index to modify inside nested functions

    def update_display():
        doc_id = files[current_index[0]][:-4]  # Remove '.txt' extension
        comparison_html = email_viewer(doc_id, processed_files_directory, unwanted_text_file_path)
        with output_area:
            clear_output(wait=True)  # Clear previous outputs
            display(HTML(comparison_html))  # Display new HTML content within the widget
        navigation_info.value = f"Document {current_index[0] + 1} of {len(files)}"

    def on_prev_button_clicked(b):
        if current_index[0] > 0:
            current_index[0] -= 1
            update_display()

    def on_next_button_clicked(b):
        if current_index[0] < len(files) - 1:
            current_index[0] += 1
            update_display()

    def on_go_button_clicked(b):
        try:
            target_id = f"{doc_id_input.value}.txt"
            target_index = files.index(target_id)
            current_index[0] = target_index
            update_display()
        except ValueError:
            with output_area:
                clear_output(wait=True)
                print("Document ID not found!")

    def on_add_button_clicked(b):
        add_unwanted_text(unwanted_text_file_path, add_text_input.value)
        add_text_input.value = ''  # Clear the input after adding

    def on_delete_button_clicked(b):
        delete_unwanted_text(unwanted_text_file_path, add_text_input.value)
        add_text_input.value = ''  # Clear the input after deleting

    def on_refresh_button_clicked(b):
        update_display()  # Refresh the current email view

    prev_button.on_click(on_prev_button_clicked)
    next_button.on_click(on_next_button_clicked)
    go_button.on_click(on_go_button_clicked)
    add_button.on_click(on_add_button_clicked)
    delete_button.on_click(on_delete_button_clicked)
    refresh_button.on_click(on_refresh_button_clicked)

    update_display()  # Initial display update

    navigation_buttons = HBox([prev_button, navigation_info, next_button])
    manage_text_buttons = HBox([add_button, delete_button, refresh_button])
    manage_text_area = VBox([add_text_input, manage_text_buttons])
    search_box = HBox([doc_id_input, go_button])
    interface = VBox([search_box, navigation_buttons, manage_text_area, output_area])

    return interface

