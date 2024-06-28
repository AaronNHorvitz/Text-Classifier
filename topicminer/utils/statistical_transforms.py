
#topicminer/utils/statistical_transformation.py

import pandas as pd
from typing import Tuple, List

# Machine Learning and topic modeling
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from topicminer.utils.email_text_processing import read_json_to_dataframe

from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Local configuration and settings
from ..config import (
    PROCESSED_TEXT_COL, 
    TOKEN_FILTER_NO_BELOW, 
    TOKEN_FILTER_NO_ABOVE,
    CATEGORIES_COL
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

    # Create a Gensim Dictionary object
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=TOKEN_FILTER_NO_BELOW, no_above=TOKEN_FILTER_NO_ABOVE)

    # Create a bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus

def upsample_classes(X, y):
    """
    Upsamples the minority classes in a dataset to have the same number of samples as the largest class.

    Parameters:
        X (array-like): 2D array containing features for each sample.
        y (array-like): Array containing class labels for each sample.

    Returns:
        tuple: A tuple containing:
            - X_resampled (array-like): The feature array after upsampling.
            - y_resampled (array-like): The target array after upsampling.
    """
    # Creating a DataFrame from numpy arrays
    df = pd.DataFrame(X)  # Convert feature array to DataFrame for manipulation
    df['target'] = y      # Append the target class labels as a new column in DataFrame

    # Find the largest class size
    max_size = df['target'].value_counts().max()  # Determine the maximum class size

    # Container for upsampled data
    upsampled_data = []

    # Upsample each class to the largest class size
    for category, group in df.groupby('target'):
        upsampled_group = resample(group, 
                                   replace=True,       # Sample with replacement
                                   n_samples=max_size, # Set the number of samples to match the largest class
                                   random_state=42)    # Ensure reproducibility
        upsampled_data.append(upsampled_group)  # Append the upsampled data

    # Concatenate all upsampled groups
    upsampled_df = pd.concat(upsampled_data)

    # Shuffle to mix class rows
    upsampled_df = upsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract features and target from the upsampled DataFrame
    y_resampled = upsampled_df['target'].values  # Extract target values
    X_resampled = upsampled_df.drop('target', axis=1).values  # Drop the target column and get the rest

    return X_resampled, y_resampled


def data_split_utility(test_size=0.33):
    """
    Splits the data into training and testing sets, applies TF-IDF vectorization, and balances the training set.

    Parameters:
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing:
            - X_train_bal (array-like): The balanced training feature array.
            - X_test (array-like): The testing feature array.
            - y_train_bal (array-like): The balanced training target array.
            - y_test (array-like): The testing target array.
    """
    # Load and prepare data
    emails_df = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL, CATEGORIES_COL])
    
    # Apply TF-IDF vectorization to the email text
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    features = tfidf.fit_transform(emails_df[PROCESSED_TEXT_COL]).toarray()
    labels = emails_df[CATEGORIES_COL]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    
    # Balance the classes in the training set
    X_train_bal, y_train_bal = upsample_classes(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test 



