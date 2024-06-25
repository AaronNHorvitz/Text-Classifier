
#topicminer/utils/statistical_transformation.py

import pandas as pd
from typing import Tuple, List

# Machine Learning and topic modeling
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from topicminer.utils.email_text_processing import read_json_to_dataframe

# Local configuration and settings
from ..config import (
    PROCESSED_TEXT_COL, 
    TOKEN_FILTER_NO_BELOW, 
    TOKEN_FILTER_NO_ABOVE
    )


def create_tfidf_corpus(corpus):
    """
    Applies TF-IDF transformation to a given corpus using a pre-defined dictionary.

    Parameters:
    ----------
    corpus : list
        The corpus to be transformed, typically as a list of bag-of-words (word_id, word_frequency) tuples.

    Returns:
    -------
    list
        A corpus transformed by TF-IDF weights.
    """
    # Initialize model
    tfidf_model = TfidfModel(corpus)

    # Apply transformation  
    corpus_tfidf = tfidf_model[corpus]  
    return corpus_tfidf

def prepare_corpus_and_dictionary() -> Tuple[Dictionary, List[List[Tuple[int, int]]]]:
    """
    Prepares a dictionary and corpus from the provided DataFrame column for use in topic modeling,
    with options to filter tokens by document frequency.

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

    emails_df = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL])

    if PROCESSED_TEXT_COL not in emails_df.columns:
        raise ValueError(
            f"Column {PROCESSED_TEXT_COL} does not exist in the DataFrame. Please change the reference in the config.py file."
        )

    # Prepare texts
    texts = [doc.split() for doc in emails_df[PROCESSED_TEXT_COL]]

    print(texts)
    # Create a Gensim Dictionary object
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=TOKEN_FILTER_NO_BELOW, no_above=TOKEN_FILTER_NO_ABOVE)

    # Create a bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus



