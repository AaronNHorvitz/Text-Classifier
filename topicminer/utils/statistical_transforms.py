# text_transformations.py

import pandas as pd
from typing import Tuple, List

# Machine Learning and topic modeling
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

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



