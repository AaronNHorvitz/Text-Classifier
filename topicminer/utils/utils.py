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


