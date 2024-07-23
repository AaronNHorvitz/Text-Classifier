# topicminer/models/classification_models/random_forest.py
"""
This module defines and manages operations related to random forest classification, 
including model training, parameter optimization, and performance evaluation.

It encapsulates functions and classes to facilitate:
- Hyperparameter tuning using Bayesian optimization.
- Training RandomForestClassifier models on text data.
- Evaluating models using various metrics like accuracy, F1-score, precision, and recall.
- Processing and transforming textual data into feature vectors.

Classes
-------
- `AutoLabelEncoder` : Manages the encoding of categorical labels automatically.

Functions
---------
- `rf_cv` : Conducts cross-validation for a RandomForestClassifier with specified parameters.
- `optimize_rf` : Optimizes hyperparameters for RandomForestClassifier using Bayesian optimization.
- `train_rf_classifier` : Trains a RandomForestClassifier with the specified parameters on balanced data.
- `score_rf_classifier` : Evaluates the performance of a RandomForestClassifier on training and test datasets.

Utilities
---------
- Uses scikit-learn for model training and evaluation.
- Employs BayesianOptimization from the `bayes_opt` library for hyperparameter tuning.
- Handles label encoding and dataset splitting with custom utilities from the `topicminer.utils` package.

Example
-------
How to train and evaluate a RandomForest model with optimized parameters:

```python
from topicminer.models.classification_models.random_forest import optimize_rf, train_rf_classifier, score_rf_classifier

# Example data preparation
X, y = load_data()  # Custom function to load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize hyperparameters
optimized_params = optimize_rf(X_train, y_train)

# Train classifier
rf_classifier = train_rf_classifier(X_train, y_train, **optimized_params)

# Evaluate classifier
score_rf_classifier(X_train, X_test, y_train, y_test, rf_classifier)
"""

# Standard Libraries
import os
import sys
from collections import Counter

# Data manipulation and numerical libraries
import numpy as np
import pandas as pd

# Visualization library
import matplotlib.pyplot as plt

# Machine Learning utilities from scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# Bayesian Optimization library
from bayes_opt import BayesianOptimization

# Custom utilities for data processing and transformations
sys.path.append(
    os.path.dirname(os.getcwd())
)  # Adjust path to include the top-level directory
from topicminer.utils.email_text_processing import read_json_to_dataframe
from topicminer.utils.statistical_transforms import (
    train_test_split_utility,
    upsample_classes,
    encode_labels,
)
from topicminer.config.config import PROCESSED_TEXT_COL, CATEGORIES_COL, TEST_SIZE

# Configuration for suppressing warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class AutoLabelEncoder:
    """
    Provides an automatic mechanism for encoding and decoding label data, 
    simplifying the handling of categorical labels for machine learning models. 
    This class is a wrapper around the `LabelEncoder` from scikit-learn, adding 
    functionality to automatically check if labels are numeric and only transform 
    them if they are not.

    Attributes
    ----------
    encoder : LabelEncoder
        A `LabelEncoder` object from scikit-learn used for encoding label data.
    is_encoded : bool
        Flag indicating whether the encoder has been fitted and labels have been 
        transformed.

    Methods
    -------
    fit_transform(y)
        Fits the label encoder and transforms labels to numeric form if they are 
        not already numeric.
    transform(y)
        Transforms labels using the fitted encoder if they have been previously 
        encoded.
    inverse_transform(y)
        Converts numeric labels back to their original form if they have been 
        previously encoded.
    is_numeric(y)
        Static method to check if the label data is numeric.

    Examples
    --------
    >>> labels = ['apple', 'banana', 'apple', 'orange']
    >>> encoder = AutoLabelEncoder()
    >>> numeric_labels = encoder.fit_transform(labels)
    >>> print(numeric_labels)
    [0 1 0 2]
    >>> original_labels = encoder.inverse_transform(numeric_labels)
    >>> print(original_labels)
    ['apple', 'banana', 'apple', 'orange']
    """

    def __init__(self):
        self.encoder = LabelEncoder()
        self.is_encoded = False

    def fit_transform(self, y):
        """
        Fit label encoder and transform labels to numeric format if not already numeric.

        Parameters
        ----------
        y : array-like
            The label data to encode. Can be any sequence-like object that can 
            be converted to an array.

        Returns
        -------
        ndarray
            Array of transformed labels in numeric format.

        Notes
        -----
        If the labels are already numeric, this method does not modify them.
        The encoder remembers the labels and can be used later for inverse transformations.
        """
        if not self.is_numeric(y):
            y = self.encoder.fit_transform(y)
            self.is_encoded = True
        return y

    def transform(self, y):
        """
        Transform labels using the already fitted encoder.

        Parameters
        ----------
        y : array-like
            Labels to transform. Must be the same as or a subset of the labels
            used during fit_transform.

        Returns
        -------
        ndarray
            Array of transformed labels in numeric format if previously encoded;
            otherwise, returns the input array as is.

        Raises
        ------
        ValueError
            If trying to transform without first calling `fit_transform`.
        """
        if self.is_encoded:
            return self.encoder.transform(y)
        return y

    def inverse_transform(self, y):
        """
        Convert numeric labels back to original labels if they were previously transformed.

        Parameters
        ----------
        y : array-like
            Numeric labels to convert back to original labels.

        Returns
        -------
        ndarray
            Array of original labels.

        Raises
        ------
        ValueError
            If trying to inverse transform without first calling `fit_transform`.
        """
        if self.is_encoded:
            return self.encoder.inverse_transform(y)
        return y

    @staticmethod
    def is_numeric(y):
        """
        Determine if the label data is already in numeric format.

        Parameters
        ----------
        y : array-like
            Label data to check.

        Returns
        -------
        bool
            True if the data type of the labels is an integer type; False otherwise.
        """
        return issubclass(y.dtype.type, np.integer)

def rf_cv(params, data, targets, scoring="f1"):
    """
    Conducts cross-validation for a RandomForestClassifier with specified parameters.

    Parameters
    ----------
    params : dict
        Parameters for the RandomForestClassifier. Keys should include 'n_estimators', 'max_depth',
        and 'min_samples_split'.
    data : array-like of shape (n_samples, n_features)
        Training data.
    targets : array-like of shape (n_samples,)
        Target labels.
    scoring : str, optional
        The scoring method to use ('accuracy', 'roc_auc', 'precision', 'recall', 'f1').
        Defaults to 'f1'.

    Returns
    -------
    float
        Mean cross-validation score according to the specified scoring method.

    Raises
    ------
    ValueError
        If an unsupported scoring method is provided.

    Examples
    --------
    >>> params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2}
    >>> data = np.array([[1, 2], [3, 4]])
    >>> targets = np.array([0, 1])
    >>> rf_cv(params, data, targets, scoring='f1')
    0.95

    Notes
    -----
    The function uses 4-fold cross-validation by default.
    """
    # Define the estimator with parameters received
    estimator = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        random_state=42,
    )

    # Check if a custom scoring function needs to be used
    if scoring not in ["accuracy", "roc_auc"]:
        # Create a scorer from the make_scorer function
        if scoring == "precision":
            scorer = make_scorer(precision_score, average="weighted")
        elif scoring == "recall":
            scorer = make_scorer(recall_score, average="weighted")
        elif scoring == "f1":
            scorer = make_scorer(f1_score, average="weighted")
        else:
            raise ValueError("Unsupported scoring method")
    else:
        scorer = scoring  # Use predefined scoring strings that sklearn recognizes

    # Return the mean of cross-validation scores
    return np.mean(cross_val_score(estimator, data, targets, scoring=scorer, cv=4))


def optimize_rf(data, targets, scoring="f1"):
    """
    Optimizes hyperparameters for a RandomForestClassifier using Bayesian optimization.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Training data.
    targets : array-like of shape (n_samples,)
        Target labels.
    scoring : str, optional
        The scoring method to use ('accuracy', 'roc_auc', 'precision', 'recall', 'f1').
        Defaults to 'f1'.

    Returns
    -------
    dict
        The result of the Bayesian optimization process, including the best parameters found and
        the corresponding best score.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> targets = np.array([0, 1])
    >>> best_params = optimize_rf(data, targets, scoring='f1')
    >>> print(best_params)

    Notes
    -----
    The function utilizes the BayesianOptimization library to perform optimization over
    specified parameter bounds. The optimization aims to maximize the cross-validation score
    of a RandomForest model as calculated by the `rf_cv` function.

    """
    param_bounds = {
        "n_estimators": (10, 500),
        "max_depth": (3, 20),
        "min_samples_split": (2, 50),
    }

    optimizer = BayesianOptimization(
        f=lambda n_estimators, max_depth, min_samples_split: rf_cv(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            },
            data,
            targets,
            scoring=scoring,
        ),
        pbounds=param_bounds,
        random_state=1,
        verbose=2,
    )

    optimizer.maximize(init_points=2, n_iter=10)

    return optimizer.max


def train_rf_classifier(
    X_train_bal, y_train_bal, n_estimators, max_depth, min_samples_split
):
    """
    Trains a RandomForestClassifier with the specified parameters.

    Parameters
    ----------
    X_train_bal : array-like of shape (n_samples, n_features)
        The training input samples, ideally balanced.
    y_train_bal : array-like of shape (n_samples,)
        The target labels for the training input samples.
    n_estimators : int
        The number of trees in the forest.
    max_depth : int
        The maximum depth of the tree.
    min_samples_split : int
        The minimum number of samples required to split an internal node.

    Returns
    -------
    RandomForestClassifier
        The trained RandomForest classifier.

    Examples
    --------
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_train = np.array([0, 1, 1])
    >>> rf_model = train_rf_classifier(X_train, y_train, 100, 15, 2)
    >>> print(rf_model)

    Notes
    -----
    This function initializes and trains a RandomForestClassifier from sklearn's ensemble
    module with the provided hyperparameters. The training process fits the model on the
    balanced dataset specified.
    """
    # Initialize the RandomForestClassifier with provided parameters
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=4,
        random_state=42,
    )

    # Fit the model on the balanced training data
    rf_clf.fit(X_train_bal, y_train_bal)

    return rf_clf


def score_rf_classifier(X_train_bal, X_test, y_train_bal, y_test, rf_clf):
    """
    Evaluates the performance of a RandomForestClassifier on both training and test datasets.

    Parameters
    ----------
    X_train_bal : array-like of shape (n_samples_train, n_features)
        Balanced training input samples.
    X_test : array-like of shape (n_samples_test, n_features)
        Test input samples.
    y_train_bal : array-like of shape (n_samples_train,)
        Target labels for the balanced training data.
    y_test : array-like of shape (n_samples_test,)
        Target labels for the test data.
    rf_clf : RandomForestClassifier
        A trained RandomForestClassifier instance.

    Returns
    -------
    None
        Outputs the accuracy, precision, recall, and F1-score for both training and test sets to the console.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    >>> score_rf_classifier(X_train, X_test, y_train, y_test, rf_clf)

    Notes
    -----
    This function predicts target values using the provided RandomForestClassifier and evaluates
    these predictions against the true labels, printing the accuracy, precision, recall, and F1-score.
    It separately evaluates and prints these metrics for both the training and test datasets.
    """
    # Predict and evaluate on the training set
    y_train_pred = rf_clf.predict(X_train_bal)
    train_accuracy = accuracy_score(y_train_bal, y_train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train_bal, y_train_pred, average="weighted"
    )

    # Predict and evaluate on the test set
    y_test_pred = rf_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="weighted"
    )

    print("==== Training Metrics ====")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1-Score: {train_f1:.4f}")

    print("\n==== Testing Metrics ====")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
