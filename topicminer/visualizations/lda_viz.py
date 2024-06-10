"""
-------------------------------------------------------------------------------
File: visualizations.py
Author: Aaron Noah Horvitz
Last Revision Date: 6/10/2024

Description:
-----------
This visualizations file is part of the TopicMiner project, aimed at processing
and analyzing Outlook emails saved as .txt files. It includes functions to
visualize data using the pre-packaged pyLDAvis library. The visualizations help
in understanding the distribution and relationship of topics within the email data.
These scripts are executed within a Jupyter Notebook environment to leverage
interactive visualization capabilities.

Key Functions:
- make_lda_viz: Executes the pyLDAvis visualization in a Jupyter Notebook, providing
  an interactive interface to explore the topics modeled by LDA.

Usage:
-----
This script is intended to be used within a Jupyter Notebook to benefit from the
interactive widgets and immediate display of processed results. Ensure that the
Jupyter Notebook has access to the required Python modules and the TopicMiner
utilities.

-------------------------------------------------------------------------------
"""

# Standard library imports
import pandas as pd

# Machine learning and topic modeling
from gensim.corpora import Dictionary
import pyLDAvis.gensim_models as gensimvis
from topicminer.utils import prepare_corpus_and_dictionary

def make_lda_viz(
        df_emails: pd.DataFrame, lda_model_tfidf, processed_text_col: str = "processed_text"
        ):
    """
    Produces the pyLDAvis visualization in a Jupyter Notebook, facilitating the exploration
    of topics extracted via LDA models.

    Parameters:
    ----------
    df_emails : pd.DataFrame
        DataFrame containing the documents to be categorized, where each row represents a document.
    processed_text_col : str, optional
        The name of the column in 'df_emails' that contains the preprocessed text suitable for topic modeling.
    lda_model_tfidf : LdaMulticore Object
        The trained LDA model from which the visualization is generated.

    Returns:
    -------
    lda_visualization : pyLDAvis._prepare.PreparedData
        An interactive PyLDAvis visualization object ready to be displayed in a Jupyter Notebook.

    Examples:
    --------
    >>> df_emails = pd.DataFrame({'processed_text': ["text about health", "text about finance"]})
    >>> lda_model = train_some_lda_model(df_emails['processed_text'])
    >>> lda_viz = make_lda_viz(df_emails, lda_model, 'processed_text')
    >>> pyLDAvis.display(lda_viz)

    Note:
    -----
    Ensure the lda_model_tfidf is properly trained and the DataFrame contains the specified column with
    preprocessed text data.
    """
    # Prepare the dictionary and corpus from the DataFrame using the specified utility function
    dictionary, corpus = prepare_corpus_and_dictionary(
        df_emails, processed_text_col
        )

    # Create pyLDAvis object for display
    lda_visualization = gensimvis.prepare(
        lda_model_tfidf, corpus, dictionary, sort_topics=False
        )

    return lda_visualization
