"""
Topic Modeling Toolkit for TopicMiner

topicminer/models/topic_models/__init__.py

This module initializes the topic modeling package for the TopicMiner project, offering streamlined access to various
models and functions specifically designed for discovering and managing topics within large text corpora. This package
supports various methods of topic extraction and management, enhancing the analysis of unstructured text data.
"""

from .lda_model import (
    train_lda_model,
    extract_topics,
    enrich_dataframe,
    train_models_and_find_optimal,
)
