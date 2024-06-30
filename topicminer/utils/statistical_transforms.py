# topicminer/utils/statistical_transformation.py
"""
This module (`statistical_transforms.py`) provides various statistical and machine learning utility functions for handling and transforming data. It includes functionality for data preprocessing, text analysis, feature extraction, data resampling, and model preparation. It is designed to facilitate the preprocessing and handling of text data for applications in topic mining and classification.

Functions:
- `suppress_warnings()`: Suppresses various warnings during runtime to clean up output.
- `create_tfidf_corpus()`: Transforms text data into a TF-IDF weighted vector format.
- `prepare_corpus_and_dictionary()`: Prepares a Gensim dictionary and corpus from textual data for topic modeling.
- `encode_labels()`: Converts categorical string labels into integers.
- `upsample_classes()`: Balances class distribution in a dataset via various resampling techniques.
- `train_test_split_utility()`: Splits the dataset into training and testing sets and optionally applies upsampling.

This module makes extensive use of libraries such as pandas, NumPy, scikit-learn, imbalanced-learn, and Gensim to perform data handling, machine learning preprocessing, and text processing tasks.

Dependencies:
- pandas: For data manipulation and analysis.
- numpy: For numerical operations on array-based data.
- scikit-learn: For machine learning model preparation and evaluation.
- imbalanced-learn: For dealing with imbalanced data via resampling methods.
- Gensim: For text processing and topic modeling.

Example:
    >>> from topicminer.utils import statistical_transforms as st
    >>> df = st.read_json_to_dataframe('email_data.json')
    >>> X_train, X_test, y_train, y_test = st.train_test_split_utility(upsampling_method='smote')
    >>> print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
"""
# Standard library imports
import warnings
from typing import Tuple, List

# Data handling
import pandas as pd
import numpy as np

# Machine Learning: General
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state, resample
from sklearn.preprocessing import LabelEncoder

# Machine Learning: Imbalanced data handling
from imblearn.over_sampling import SMOTE, ADASYN

# Text processing and topic modeling
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# Local utilities and configurations
from topicminer.utils.email_text_processing import read_json_to_dataframe
from ..config import (
    PROCESSED_TEXT_COL,
    TOKEN_FILTER_NO_BELOW,
    TOKEN_FILTER_NO_ABOVE,
    CATEGORIES_COL,
    TEST_SIZE,
    RANDOM_STATE,
    WORD_COUNT_COL,
    CHARACTER_COUNT_COL,
    TOKEN_COUNT_COL,
)


def suppress_warnings():
    """
    Suppresses specific warnings in Python scripts, particularly from the sklearn module.

    This function is useful for hiding warnings that are not critical and might clutter the output,
    especially when using sklearn for training machine learning models where some warnings
    about convergence or deprecations may not be immediately actionable.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    To use this function, simply call it before running the portion of your code where warnings are expected:

    >>> suppress_warnings()
    >>> # Your code here where warnings are expected

    This function cat be placed at the start of any script or before any specific operations where warnings might be
    distracting or irrelevant to the current analysis. This ensures that the output remains clean and focused on the
    essential information.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

    # Suppress all other warnings
    warnings.filterwarnings("ignore")


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
    Upsample minority classes in a dataset to address class imbalance using the specified method.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix where n_samples is the number of samples and n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target vector containing class labels for each sample.
    method : str, optional
        The method used for upsampling. Options include:
        - "duplicate": Duplicates existing samples to balance class distributions.
        - "smote": Synthetic Minority Over-sampling Technique; generates synthetic samples.
        - "adasyn": Adaptive Synthetic Sampling; similar to SMOTE but with a focus on harder samples.
        By default, None, which will use "duplicate" if no method is specified.

    Returns
    -------
    X_resampled : ndarray, shape (n_resampled_samples, n_features)
        The resampled feature matrix.
    y_resampled : ndarray, shape (n_resampled_samples,)
        The resampled target vector.

    Raises
    ------
    ValueError
        If resampling with SMOTE or ADASYN fails due to insufficient neighbors.

    Notes
    -----
    This function first checks if the target data `y` needs to be encoded from strings to integers,
    performs the encoding if necessary, and then applies the chosen resampling method.
    If the chosen method fails due to parameter issues (e.g., not enough neighbors for SMOTE),
    it attempts to adjust the parameters or reverts to duplicating samples.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
    ... n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1,
    ... n_samples=100, random_state=42)
    >>> X_resampled, y_resampled = upsample_classes(X, y, method='smote')
    >>> print(np.bincount(y_resampled))
    array([90, 90])
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


def train_test_split_utility(upsampling_method=None):
    """
    Splits the dataset into training and testing sets and optionally applies upsampling to address class imbalance.

    This function integrates text features with additional numerical features extracted from emails, such as word count,
    character count, and token count, creating a comprehensive feature set for model training.

    Parameters
    ----------
    upsampling_method : str, optional
        The method used for upsampling the minority class in the training dataset. Options include:
        - 'duplicate': Duplicates samples in minority classes.
        - 'smote': Synthetic Minority Over-sampling Technique.
        - 'adasyn': Adaptive Synthetic Sampling Approach.
        If None, no upsampling is applied. Default is None.

    Returns
    -------
    tuple
        A tuple of four elements containing:
        - X_train_bal (ndarray): The feature matrix for the training data after optional upsampling.
        - X_test (ndarray): The feature matrix for the testing data.
        - y_train_bal (ndarray): The target vector for the training data after optional upsampling.
        - y_test (ndarray): The target vector for the testing data.

    Notes
    -----
    The function reads data from a predefined source which is assumed to include specific columns for text and labels, along with
    additional numeric features. It combines text features processed through TF-IDF vectorization with these numeric features
    to form the final feature set used for model training and testing.

    Examples
    --------
    >>> X_train_bal, X_test, y_train_bal, y_test = train_test_split_utility(upsampling_method='smote')
    >>> print(f"Training features shape: {X_train_bal.shape}")
    >>> print(f"Test features shape: {X_test.shape}")
    """
    # Load and prepare data
    emails_df = read_json_to_dataframe(
        columns=[
            PROCESSED_TEXT_COL,
            CATEGORIES_COL,
            WORD_COUNT_COL,
            CHARACTER_COUNT_COL,
            TOKEN_COUNT_COL,
        ]
    )

    # Create initial TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_features = tfidf.fit_transform(emails_df[PROCESSED_TEXT_COL])

    # Extract additional features
    additional_features = emails_df[
        ["word_count", "character_count", "token_count"]
    ].values

    # Combine TF-IDF features with additional features
    features = hstack([tfidf_features, additional_features]).toarray()

    # Obtain and encode labels
    labels = emails_df[CATEGORIES_COL]
    if not issubclass(labels.dtype.type, np.integer):
        labels = encode_labels(labels)

    # Split the data
    random_state = check_random_state(RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=random_state
    )

    print(
        f"Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    # Upsample Data
    X_train_bal, y_train_bal = X_train, y_train
    if upsampling_method:
        X_train_bal, y_train_bal = upsample_classes(
            X_train, y_train, method=upsampling_method
        )

    return X_train_bal, X_test, y_train_bal, y_test
