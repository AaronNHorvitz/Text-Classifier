#TODO: resolve unwanted_texts_filepath function issue for import into current funcitons and widgets.py 
#TODO: fix text_pruner so it doesn't show multiple emails.
# 
# 
#  
""" 
-------------------------------------------------------------------------------
File: widgets.py
Author: Aaron Noah Horvitz
Last Revision Date: 6/10/2024

Description:
-----------
Interactive Widgets Module for TopicMiner

This module provides interactive Jupyter Notebook widgets for the TopicMiner project, facilitating the review and management of email data.
It includes functionalities for displaying email texts with highlighted unwanted texts, managing unwanted texts, and navigating through email documents stored in directories.

Features:
---------
- email_viewer: Displays an email with highlighted unwanted text and provides a cleaned and normalized text comparison in an HTML table format.
- email_viewer_widget: Creates an interactive widget for navigating and managing emails within a Jupyter Notebook.

Dependencies:
-------------
- Standard Libraries: os
- Third-party Libraries: numpy, pandas, IPython, ipywidgets
- Natural Language Processing: nltk
- TopicMiner Utilities: Various utilities from the 'text_processing' and 'email_processing' modules within the TopicMiner project.

Usage:
------
These widgets are intended to be used within Jupyter Notebooks to allow users to interactively manage and review email data. They provide tools for visual data inspection and text cleaning in a user-friendly interface.

Example:
--------
To initialize and display an email viewer widget within a Jupyter Notebook, you might use:

>>> from widgets import email_viewer_widget
>>> email_widget = email_viewer_widget('/path/to/email/files', '/path/to/unwanted_texts.json')
>>> display(email_widget)

Notes:
------
Ensure that the NLTK data path is correctly set if using NLTK resources. These widgets rely on the interactive capabilities of IPython and Jupyter environments; therefore, they are not suitable for execution in non-interactive Python script environments.
"""

from pathlib import Path
from ipywidgets import Button, Text, VBox, HBox, Output, Label, HTML
import json
import os



# Standard library imports
import os
import json

# Third-party library imports
import numpy as np
import pandas as pd
from IPython.display import display, HTML, clear_output
from ipywidgets import Button, HBox, VBox, Output, Layout, Text, Label

# Natural Language Processing tools from NLTK
from nltk import data

# Local imports from the TopicMiner project
from topicminer.utils.email_text_processing import (
    read_text_file,
    preprocess_text,
    parse_top_email_from_chain,
    clean_email_body_text,
    add_unwanted_email_text,
    delete_unwanted_email_text,
    load_unwanted_email_text,
    save_unwanted_texts,
    strip_top_email
)

# Add the path to the NLTK data directory if the data is not found locally
data.path.append("./topicminer/data/nltk_data")

from ..config import UNWANTED_TEXTS_FILE, RAW_DATA_DIR

# # Use the imported path in your module
# with open(UNWANTED_TEXTS_FILE, 'r') as file:
#     data = json.load(file)


def email_viewer(
    doc_id: str,
    data_path: str,
    unwanted_text_filepath: str = "./data/unwanted_texts/unwanted_texts.json",
) -> str:
    """
    Displays an email text comparison in a formatted HTML table, including the original email with highlighted unwanted text,
    the cleaned email text with unwanted text removed, and the preprocessed text for modeling.

    Parameters
    ----------
    doc_id : str
        Document ID for the email, which corresponds to the filename without the '.txt' extension.
    data_path : str
        Path to the directory containing the email files. Each file should be a text file (.txt) named by its document ID.
    unwanted_text_filepath : str
        Path to the JSON file containing a list of unwanted text strings to be highlighted and removed from the email content.

    Returns
    -------
    str
        A string containing HTML content that represents a table with three columns: 'Original Email (Highlighted Unwanted Text)',
        'Removed Unwanted Text', and 'Normalized Text'. Each cell contains respective content styled and formatted for clear presentation.

    Raises
    ------
    Exception
        Catches and prints exceptions related to file handling or processing errors.

    Examples
    --------
    >>> email_viewer('sample_doc_id', '/path/to/email/directory', '/path/to/unwanted_texts.json')

    Notes
    -----
    This function reads the specified email file based on the document ID, processes it to highlight and remove unwanted texts,
    and then applies text preprocessing techniques like tokenization and lemmatization. The results are formatted into an HTML table
    for visual comparison in a Jupyter Notebook or other environments that support HTML rendering.
    """

    # Construct full path to the email file
    text_file_path = os.path.join(data_path, f"{doc_id}.txt")

    # Load unwanted texts for highlighting
    unwanted_texts, unwanted_texts_filepath = load_unwanted_email_text(unwanted_text_filepath)

    try:
        # Retrieve and parse the original email text
        text_file_contents = read_text_file(text_file_path, word_wrap_limit=50)
        email_recipient, email_from, email_date, email_subject, email_body = (
            parse_top_email_from_chain(text_file_contents)
        )
        reconstructed_email = f"To: {email_recipient}\nFrom: {email_from}\nSent: {email_date}\n\nSubject: {email_subject}\n\n{email_body}"
        reconstructed_email = "".join(reconstructed_email)

        # Highlight unwanted text in orange
        highlighted_email_body = reconstructed_email
        for phrase in unwanted_texts:
            highlighted_email_body = highlighted_email_body.replace(
                phrase, f"<mark style='background-color: orange;'>{phrase}</mark>"
            )

        # Remove unwanted text to prepare a cleaned email.
        cleaned_email = clean_email_body_text(reconstructed_email, unwanted_texts)
        cleaned_email = "".join(cleaned_email)

        cleaned_email_body = clean_email_body_text(email_body, unwanted_texts)
        normalized_text = preprocess_text(cleaned_email_body)
        normalized_text = "".join(normalized_text)

        # Create HTML content for display with customizable font size and type
        comparison_html = f"""
        <style>
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}  /* Ensure content is top-aligned */
            th {{ background-color: #f2f2f2; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; text-align: left; margin: 0; max-width: 600px; font-family: 'Arial', sans-serif; font-size: 14px; }}
            mark {{ background-color: orange; }}
            .text-container {{ text-align: left; overflow-wrap: break-word; min-height: 400px; }}  /* Adjusted for more vertical space */
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


def processed_email_viewer_widget(
    processed_files_directory: str = "./data/raw_data/processed_emails.csv",
    unwanted_text_file_path: str = "./data/unwanted_texts/unwanted_texts.json",
):
    """
    Creates an interactive Jupyter Notebook widget for viewing and managing email documents.
    It allows users to navigate through email documents stored in a specified directory,
    display them with unwanted texts highlighted, and interactively manage unwanted texts using a JSON file.

    The widget includes navigation buttons to move between emails, a text box to directly jump to a specific email by document ID,
    and buttons for adding or deleting text strings to/from an unwanted text list.

    Parameters:
    ----------
    processed_files_directory : str
        The directory path where processed email files (.txt) are stored.
    unwanted_text_file_path : str
        The path to the JSON file that stores a list of unwanted text strings used for cleaning the emails.

    Returns:
    -------
    ipywidgets.VBox
        A VBox widget containing all interactive elements including navigation buttons, document display area,
        and text management tools.

    Examples:
    --------
    Assuming `data_path` and `unwanted_texts_path` are your directory and file paths respectively:

    >>> email_widget = email_viewer_widget(data_path, unwanted_texts_path)
    >>> display(email_widget)

    This function integrates several key functionalities:
    - It reads and displays emails from text files using their document IDs.
    - It allows users to navigate through emails using 'Previous' and 'Next' buttons.
    - Users can jump to any email by entering its document ID in the 'Go to ID' text box.
    - Users can add or remove phrases from the unwanted texts list using the provided text box and corresponding buttons.
    - The 'Refresh' button can be used to reload the current email text, reflecting any changes made to the unwanted texts.

    Notes:
    -----
    Ensure that both the `processed_files_directory` and `unwanted_text_file_path` are correctly specified and accessible.
    This widget is designed to facilitate the review and cleaning process of email texts for further analysis or modeling.
    """

    files = sorted(
        [f for f in os.listdir(processed_files_directory) if f.endswith(".txt")]
    )
    if not files:
        print("No text files found in the directory.")
        return

    output_area = Output(
        layout={"border": "1px solid black", "width": "100%", "height": "500px"}
    )
    doc_id_input = Text(placeholder="Enter Document ID", description="Go to ID:")
    text_to_manage = Text(
        placeholder="Enter text to add/delete", description="Manage Text:"
    )
    go_button = Button(description="Go")
    prev_button = Button(description="Previous")
    next_button = Button(description="Next")
    add_text_button = Button(description="Add Text")
    delete_text_button = Button(description="Delete Text")
    refresh_button = Button(description="Refresh")
    navigation_info = Label()
    current_doc_label = Label()

    def update_display(index):
        nonlocal current_index  # Ensure current_index can be modified
        doc_id = files[index][:-4]  # Remove '.txt' extension
        current_doc_label.value = f"Viewing: {doc_id}"
        comparison_html = email_viewer(
            doc_id,
            processed_files_directory,
            unwanted_texts=load_unwanted_email_text(file_path=unwanted_text_file_path),
        )
        with output_area:
            clear_output(wait=True)
            display(HTML(comparison_html))
        navigation_info.value = f"Document {index + 1} of {len(files)}"
        current_index = index  # Update the current index

    def on_prev_button_clicked(b):
        if current_index > 0:
            update_display(current_index - 1)

    def on_next_button_clicked(b):
        if current_index < len(files) - 1:
            update_display(current_index + 1)

    def on_go_button_clicked(b):
        try:
            target_id = f"{doc_id_input.value}.txt"
            target_index = files.index(target_id)
            update_display(target_index)
        except ValueError:
            with output_area:
                clear_output(wait=True)
                print("Document ID not found!")

    def on_add_text_button_clicked(b):
        add_unwanted_email_text(unwanted_text_file_path, text_to_manage.value)
        text_to_manage.value = ""  # Clear the input field

    def on_delete_text_button_clicked(b):
        delete_unwanted_email_text(unwanted_text_file_path, text_to_manage.value)
        text_to_manage.value = ""  # Clear the input field

    def on_refresh_button_clicked(b):
        update_display(current_index)

    prev_button.on_click(on_prev_button_clicked)
    next_button.on_click(on_next_button_clicked)
    go_button.on_click(on_go_button_clicked)
    add_text_button.on_click(on_add_text_button_clicked)
    delete_text_button.on_click(on_delete_text_button_clicked)
    refresh_button.on_click(on_refresh_button_clicked)

    current_index = 0
    update_display(current_index)  # Initialize the display with the first document

    # Layouts
    navigation_buttons = HBox([prev_button, navigation_info, next_button])
    text_management_buttons = HBox(
        [add_text_button, delete_text_button, refresh_button, text_to_manage]
    )
    search_box = HBox([doc_id_input, go_button])
    doc_info = HBox([current_doc_label])

    left_box = VBox([search_box, navigation_buttons, doc_info])
    right_box = VBox([text_management_buttons])
    top_row = HBox(
        [left_box, right_box], layout=Layout(justify_content="space-between")
    )
    interface = VBox([top_row, output_area])

    return interface


def interactive_email_viewer_widget(
    data_path: str = None,
    unwanted_text_file_path: str = "./data/unwanted_texts/unwanted_texts.json",
):
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
    # Establish data path if none is provided
    if data_path is None:
        src_path = os.path.dirname(os.getcwd())
        data_path = os.path.join(src_path, 'data/raw_data')

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

        # Read the text file contents and parse the email components
        file_path = os.path.join(data_path, files[idx])
        # original_text = view_file(file_path)
        original_text = read_text_file(file_path)
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
        ) = parse_top_email_from_chain(original_text)

        # Load unwanted texts and preprocess the email body
        processed_text = preprocess_text(email_body)

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
                    <td class="email-view"><b>Processed Email:</b>
                    <br>{'From: ' + email_sender}
                    <br>{'Sent: ' + email_date}
                    <br>{'To: ' + email_recipient}
                    <br>{'CC: ' + email_cc}
                    <br>{'BCC: ' + email_bcc}
                    <br>{'Subject: ' + email_subject}
                    <br>{'Attachments: ' + email_attachments}
                    <br><br>{'Categories: ' + email_categories}
                    <br><br>{processed_text}</td>
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

    
def text_pruner() -> VBox:

    unwanted_texts = load_unwanted_email_text()
    
    files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".txt")])
    print(files)
    index = [0]
    
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
        output.clear_output(wait=True)  # Using wait=True to clear output just before displaying new content
        file_path = os.path.join(RAW_DATA_DIR, files[idx])
        with open(file_path, "r", encoding="utf-8") as file:
            email_content = file.read()

        # Process the email content to remove unwanted texts
        email_content_no_unwanted = email_content
        for phrase in unwanted_texts:
            highlighted_phrase = f"<mark style='background-color: red;'>{phrase}</mark>"  # Corrected typo 'backgzround-color'
            email_content = email_content.replace(phrase, highlighted_phrase)
            email_content_no_unwanted = email_content_no_unwanted.replace(phrase, "")

        # # Strip top email
        # email_content_no_unwanted = strip_top_email(email_content_no_unwanted)
        
        # Simulate processing the email content
        processed_text = preprocess_text(email_content_no_unwanted)

        with output:
            display(
                HTML(
                    f"""
                    <style>
                        .email-view {{
                            width: 32%;
                            overflow-wrap: break-word;
                            white-space: pre-wrap;
                            text-align: left;
                            padding: 10px;
                        }}
                        table {{
                            width: 100%;
                            table-layout: fixed;
                            border-collapse: collapse;
                        }}
                        td {{
                            vertical-align: top;
                            border: 1px solid #ccc;
                        }}
                        mark {{
                            background-color: yellow;
                            font-weight: bold;
                        }}
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

    # Initialize the view
    show_email(index[0])

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
            save_unwanted_texts(UNWANTED_TEXTS_FILE, unwanted_texts)
            txt_add_unwanted.value = ""
            show_email(index[0])

    def remove_text(text):
        if text in unwanted_texts:
            unwanted_texts.remove(text)
            save_unwanted_texts(UNWANTED_TEXTS_FILE, unwanted_texts)
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
