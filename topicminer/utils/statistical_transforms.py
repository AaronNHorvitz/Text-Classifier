
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

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import resample
import numpy as np


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

def upsample_classes(X, y, method='duplicate'):
    """
    Upsamples the minority classes in a dataset using the specified method to help balance the class distribution.
    
    Parameters:
        X (numpy.ndarray): Feature matrix containing the independent variables.
        y (numpy.ndarray): Target vector containing class labels.
        method (str): The method to use for upsampling. Options include:
            - 'duplicate': Duplicates existing samples in the minority classes until all classes have the same number of samples. 
              This method increases the number of samples by copying existing instances, which can lead to overfitting as it does not introduce any new information.
            - 'smote': Synthetic Minority Over-sampling Technique. Generates synthetic samples rather than duplicating existing ones. 
              This method uses k-nearest neighbors to create new, synthetic samples that are similar but not identical to existing samples, potentially adding more diversity and reducing the risk of overfitting.
            - 'adasyn': Adaptive Synthetic Sampling Approach. Similar to SMOTE but with a focus on generating samples next to the original samples that are wrongly classified using a k-nearest neighbors classifier. 
              This method aims to adaptively generate minority data points according to their distribution: more synthetic data is generated for minority class samples that are harder to learn.

    Returns:
        tuple: A tuple containing:
            - X_resampled (numpy.ndarray): The resampled feature matrix after applying the upsampling.
            - y_resampled (numpy.ndarray): The resampled target vector after applying the upsampling.
    
    Example Usage:
        >>> X_resampled, y_resampled = upsample_classes(X_train, y_train, method='smote')
        This will balance the classes in the training data using the SMOTE method.
    """
    # Check the method specified for upsampling and execute accordingly
    if method == 'duplicate':
        # Convert feature matrix X and target vector y into a DataFrame
        df = pd.DataFrame(X)
        df['target'] = y
        
        # Find the largest class size
        max_size = df['target'].value_counts().max()
        
        # Upsample each class to the size of the largest class using resampling
        upsampled_data = [resample(group, replace=True, n_samples=max_size, random_state=42)
                        for _, group in df.groupby('target')]
        
        # Concatenate upsampled data and shuffle
        upsampled_df = pd.concat(upsampled_data).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Extract features and targets from the upsampled DataFrame
        X_resampled = upsampled_df.drop('target', axis=1).values
        y_resampled = upsampled_df['target'].values

    elif method == 'smote':
        # Initialize and apply SMOTE to generate synthetic samples
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    elif method == 'adasyn':
        # Initialize and apply ADASYN to generate synthetic samples focusing on hard samples
        adasyn = ADASYN(random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

    return X_resampled, y_resampled

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def train_test_split_utility(test_size=0.33, upsampling_method=None):
    """
    Splits the dataset into training and testing sets, applies TF-IDF vectorization to the text data, 
    and optionally balances the training set using a specified upsampling method.

    Parameters:
    ----------
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.33.
    upsampling_method (str or None): The method to use for balancing the training set. Options are 'duplicate', 'smote', 'adasyn', or None. If None, no upsampling is applied. Detailed methods:
        - 'duplicate': Simple duplication of existing samples to balance class distributions.
        - 'smote': Synthetic Minority Over-sampling Technique, which synthetically generates new samples based on existing minority samples.
        - 'adasyn': Adaptive Synthetic Sampling, which generates new samples with a focus on samples that are harder to classify.

    Returns:
    -------
    tuple: A tuple containing:
        - X_train_bal (numpy.ndarray): The balanced training feature array after vectorization and optional upsampling.
        - X_test (numpy.ndarray): The testing feature array after vectorization.
        - y_train_bal (numpy.ndarray): The balanced training target array after optional upsampling.
        - y_test (numpy.ndarray): The testing target array.
    
    Example Usage:
    -------------
    >>> X_train_bal, X_test, y_train_bal, y_test = train_test_split_utility(test_size=0.25, upsampling_method='smote')
    This splits the data, applies TF-IDF, and optionally balances the training set using SMOTE.
    """
    
    # Load and prepare data
    emails_df = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL, CATEGORIES_COL])
    
    # Apply TF-IDF vectorization to the email text
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    features = tfidf.fit_transform(emails_df[PROCESSED_TEXT_COL]).toarray()
    labels = emails_df[CATEGORIES_COL]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    
    # Initialize balanced data as original data by default
    X_train_bal, y_train_bal = X_train, y_train

    # Check if upsampling is requested
    if upsampling_method:
        # Balance the classes in the training set using the specified method
        X_train_bal, y_train_bal = upsample_classes(X_train, y_train, method=upsampling_method)

    return X_train_bal, X_test, y_train_bal, y_test



