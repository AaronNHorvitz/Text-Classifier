#topicminer/models/classificaiton_models/__init__.py
"""
Classification Models for TopicMiner

topicminer/models/classification_models/__init__.py

This module initializes the classification package for the TopicMiner project, providing access to various
classification models and functions specifically designed for categorizing and analyzing text data. This package
supports a range of machine learning models, enhancing the project's ability to make predictions and derive insights
from unstructured text data.

Models included:
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Logistic Regression
- Decision Tree
- Ensemble Methods

Each model module contains functions to train, evaluate, and utilize classifiers for predictive tasks, ensuring
comprehensive coverage of typical classification needs in natural language processing applications.
"""


from .rf_classifier import *
from .xgb_classifier import *
# from .svm import *
# from .naive_bayes import *
# from .logistic_regression import *
# from .decision_tree import *
# from .ensemble import *

