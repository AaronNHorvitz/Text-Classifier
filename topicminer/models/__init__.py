"""
Models Package for TopicMiner

topicminer/models/__init__.py

This module acts as a central hub for importing and managing various machine learning and topic modeling functions within
the TopicMiner project. It provides access to high-level functionalities that encompass various aspects of model training,
topic extraction, and data enrichment across different model types including topic models and potentially other classification
or clustering models in the future.
"""

from .topic_models import (
    train_lda_model,
    extract_topics,
    enrich_dataframe,
    train_models_and_find_optimal,
)
