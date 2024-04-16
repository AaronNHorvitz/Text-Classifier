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

import pandas as pd
import numpy as np

from datetime import datetime
import textwrap


def convert_date_format(date_str):
    """
    Convert a date string from 'YYYY-MM-DD' format to 'Month DD, YYYY' format.

    Parameters:
    date_str (str): A string representing the date in 'YYYY-MM-DD' format.

    Returns:
    str: A string representing the date in 'Month DD, YYYY' format.

    Example:
    >>> convert_date_format('2011-01-13')
    'January 13, 2011'
    """
    # Check if the date string is NaN
    if pd.isna(date_str):
        return np.nan

    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    # Format the datetime object into the desired string format
    new_date_str = date_obj.strftime("%B %d, %Y")

    return new_date_str

def generate_email_text(email_date, email_recipient, email_from, email_subject, email_body):
    """
    Generates formatted email text with word-wrapped body.

    Parameters:
    email_date (str): Date of the document.
    email_recipient (str): Recipient of the email.
    email_from (str): Sender of the email.
    email_subject (str): Subject of the email.
    email_body (str): Body of the email.

    Returns:
    str: Formatted email text.
    """
    email_date = convert_date_format(email_date) if not pd.isna(email_date) else "(Unknown Date)"
    email_from = email_from if not pd.isna(email_from) else "(Unknown Sender)"
    email_recipient = email_recipient if not pd.isna(email_recipient) else "(Unknown Recipient)"
    
    # Correctly handle no subject and unknown subject cases
    if pd.isna(email_subject) or email_subject.strip() == '(NO SUBJECT)':
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

def format_and_save_emails(df, output_dir):
    """
    Formats each row in the DataFrame as an email and saves it as a text file.

    Parameters:
    df (DataFrame): DataFrame containing email data.
    output_dir (str): Directory where the email text files will be saved.
    """
    for index, row in df.iterrows():
        email_text = generate_email_text(
            row["docDate"], row["to"], row["from"], row["subject"], row["docText"]
        )
        #print(email_text)
        file_path = f"{output_dir}/{row['docID']}.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(email_text)
        print("Processed email document ID: {}".format(row["docID"]))
