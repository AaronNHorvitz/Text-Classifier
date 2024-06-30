# topicminer/utils/statistical_transformation.py

import pandas as pd
import warnings
from typing import Tuple, List

# Machine Learning and topic modeling
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from topicminer.utils.email_text_processing import read_json_to_dataframe


from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import resample
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

# Local configuration and settings
from ..config import (
    PROCESSED_TEXT_COL,
    TOKEN_FILTER_NO_BELOW,
    TOKEN_FILTER_NO_ABOVE,
    CATEGORIES_COL,
    TEST_SIZE,
)


def suppress_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
    dictionary.filter_extremes(
        no_below=TOKEN_FILTER_NO_BELOW, no_above=TOKEN_FILTER_NO_ABOVE
    )

    # Create a bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus


def encode_labels(y):
    """
    Encodes string class labels to integers.
    """
    encoder = LabelEncoder()
    return encoder.fit_transform(y)


def upsample_classes(X, y, method=None):
    """
    Upsamples the minority classes in a dataset to balance the class distribution using the specified method.

    Parameters:
    ----------
        X (numpy.ndarray): The feature matrix where each row represents a sample.
        y (numpy.ndarray or list): The target array or list where each element corresponds to the class of the samples in X.
        method (str, optional): The method to use for upsampling. Options include:
            - 'duplicate': Duplicates existing samples in the minority classes until all classes have the same number of samples.
            - 'smote': Synthetic Minority Over-sampling Technique. Generates synthetic samples rather than duplicating existing ones.
            - 'adasyn': Adaptive Synthetic Sampling Approach. Similar to SMOTE but focuses more on generating samples near the decision boundary.
            If None is provided, the function will default to 'duplicate' if upsampling is needed due to a failure in advanced methods like SMOTE or ADASYN.

    Returns:
    -------
        tuple: A tuple containing:
            - X_resampled (numpy.ndarray): The resampled feature matrix after applying the upsampling.
            - y_resampled (numpy.ndarray): The resampled target array after applying the upsampling.

    Raises:
    ------
        ValueError: If resampling with SMOTE or ADASYN fails due to configuration issues, such as not having enough neighbors.

    Notes:
    -----
        This function suppresses warnings internally to avoid clutter during processing.
        It also ensures that the target labels 'y' are properly encoded as integers if they are initially in string format.
    """
    suppress_warnings()  # Suppress warnings within this function

    # Ensure y is properly encoded as integers
    if isinstance(y[0], str):
        y = encode_labels(y)

    if method == "duplicate":
        df = pd.DataFrame(X)
        df["target"] = y
        max_size = df["target"].value_counts().max()
        upsampled_data = [
            resample(group, replace=True, n_samples=max_size, random_state=42)
            for _, group in df.groupby("target")
        ]
        upsampled_df = (
            pd.concat(upsampled_data)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        X_resampled = upsampled_df.drop("target", axis=1).values
        y_resampled = upsampled_df["target"].values
    else:
        # Initialize resampler based on the method
        resampler = (
            SMOTE(random_state=42) if method == "smote" else ADASYN(random_state=42)
        )
        try:
            X_resampled, y_resampled = resampler.fit_resample(X, y)
        except ValueError as e:
            # If initial resampling fails, dynamically adjust parameters
            min_class_size = np.bincount(y).min()
            for n_neighbors in range(1, max(2, min_class_size)):
                try:
                    resampler.set_params(**{"n_neighbors": n_neighbors})
                    X_resampled, y_resampled = resampler.fit_resample(X, y)
                    break
                except ValueError:
                    continue
            else:
                # If all else fails, revert to duplication
                return upsample_classes(X, y, method="duplicate")

    return X_resampled, y_resampled


def train_test_split_utility(test_size=0.33, upsampling_method=None, random_state=42):
    """
    Splits the dataset into training and testing sets, vectorizes the text data using TF-IDF,
    and optionally balances the training set using a specified upsampling method.

    Parameters:
    ----------
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.33.
        upsampling_method (str, optional): The method to use for balancing the training set.
            Options are 'duplicate', 'smote', 'adasyn', or None. If None, no upsampling is applied.
        random_state (int, RandomState instance or None): Controls the randomness of the training and testing data split.
            Pass an int for reproducible output across multiple function calls.

    Returns:
    -------
        tuple: A tuple containing:
            - X_train_bal (numpy.ndarray): The balanced (if upsampling_method is specified) training feature array.
            - X_test (numpy.ndarray): The testing feature array.
            - y_train_bal (numpy.ndarray): The balanced (if upsampling_method is specified) training target array.
            - y_test (numpy.ndarray): The testing target array.

    Raises:
    ------
        ValueError: If the 'upsampling_method' is not one of the expected options.

    Notes:
    -----
        The function first loads and preprocesses the email data from a predefined JSON data structure into feature
        vectors using TF-IDF vectorization. It then splits these features into training and testing sets.
        If an upsampling method is specified, it applies the selected method to balance class distribution in the training set.
    """
    random_state = check_random_state(random_state)

    # Load and prepare data
    emails_df = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL, CATEGORIES_COL])
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    features = tfidf.fit_transform(emails_df[PROCESSED_TEXT_COL]).toarray()
    labels = emails_df[CATEGORIES_COL]

    if not issubclass(labels.dtype.type, np.integer):
        labels = encode_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    # Print data shape for debugging
    print(
        f"Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    X_train_bal, y_train_bal = X_train, y_train
    if upsampling_method:
        X_train_bal, y_train_bal = upsample_classes(
            X_train, y_train, method=upsampling_method
        )

    return X_train_bal, X_test, y_train_bal, y_test
