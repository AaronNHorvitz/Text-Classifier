
import pandas as pd
import numpy as np

from tqdm import tqdm

from gensim.models import LdaMulticore
from typing import Tuple, List

from topicminer.utils import prepare_corpus_and_dictionary, create_tfidf_corpus


def train_lda_model(
    emails_df: pd.DataFrame,
    processed_text_col: str = "processed_text",
    num_topics: int = 10,
    workers: int = 8,
    chunksize: int = 2000,
    passes: int = 1,
    no_below: int = 5,
    no_above: float = 0.5,
    use_tfidf: bool = True,
    batch: bool = False,
    alpha: str = "auto",
    eta=None,
    decay: float = 0.5,
    offset: float = 1,
    eval_every: int = 10,
    iterations: int = 50,
    gamma_threshold: float = 0.001,
    random_state: int = 100,
    minimum_probability: float = 0.01,
    minimum_phi_value: float = 0.01,
    per_word_topics: bool = False,
    dtype: type = np.float32,
) -> LdaMulticore:
    """
    Trains an LDA (Latent Dirichlet Allocation) model using the Gensim library with support for multicore processing.
    It prepares the corpus and optionally applies TF-IDF transformation before training the model. This function is configured
    with several parameters to customize the training process according to specific needs.

    Parameters:
    ----------
    emails_df : pd.DataFrame
        DataFrame containing the emails with a column of preprocessed text.
    processed_text_col : str
        Name of the column in `emails_df` that contains preprocessed text for topic modeling.
    num_topics : int
        The number of latent topics to extract from the corpus.
    workers : int
        The number of worker processes to use for parallelization. Capped at 8.
    chunksize : int
        The number of documents to process at a time in the training algorithm.
    passes : int
        The number of full passes over the corpus during training.
    no_below : int
        Minimum number of documents a token must appear in to be included in the corpus.
    no_above : float
        Maximum proportion of documents a token can appear in to be included in the corpus.
    use_tfidf : bool
        Flag to determine if TF-IDF transformation should be applied to the corpus before training.
    batch : bool
        Whether to use all documents in each training chunk.
    alpha : str
        The hyperparameter affecting the sparsity/thickness of the topics.
    eta : optional
        The eta hyperparameter, affecting topic-word density.
    decay : float
        Factor for learning rate decay.
    offset : float
        Hyperparameter that offsets early iterations.
    eval_every : int
        The interval at which the model parameters should be updated.
    iterations : int
        Maximum number of iterations over each document.
    gamma_threshold : float
        Convergence threshold for gamma updates.
    random_state : int
        Seed for random number generation for reproducibility.
    minimum_probability : float
        Minimum probability cutoff to consider a topic in a document.
    minimum_phi_value : float
        Minimum probability cutoff to consider a word in a topic.
    per_word_topics : bool
        If True, computes a list of topics for each word.
    dtype : type
        Data type to use during calculations.

    Returns:
    -------
    LdaMulticore
        The trained LDA model.

    Examples:
    --------
    >>> lda_model = train_lda_model(emails_df, 'processed_text', num_topics=5, passes=15, workers=4)
    """
    if workers > 8:
        workers = 8
        print("The maximum number of workers allowed is 8. Setting workers to 8.")

    dictionary, corpus = prepare_corpus_and_dictionary(
        emails_df, processed_text_col, no_below, no_above
    )

    if use_tfidf:
        corpus = create_tfidf_corpus(corpus)

    lda_model = LdaMulticore(
        corpus=corpus,
        num_topics=num_topics,
        id2word=dictionary,
        workers=workers,
        chunksize=chunksize,
        passes=passes,
        batch=batch,
        alpha=alpha,
        eta=eta,
        decay=decay,
        offset=offset,
        eval_every=eval_every,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        random_state=random_state,
        minimum_probability=minimum_probability,
        minimum_phi_value=minimum_phi_value,
        per_word_topics=per_word_topics,
        dtype=dtype,
    )

    if passes > 1:
        for pass_idx in tqdm(range(1, passes), desc="Training LDA Model"):
            lda_model.update(corpus)

    return lda_model


def extract_topics(lda_model: LdaMulticore, corpus: List[List[tuple]]) -> List[str]:
    """
    Extracts and formats the dominant topics and their corresponding probabilities from the LDA model for each document in the corpus.

    Parameters:
    ----------
    lda_model : LdaModel or LdaMulticore
        The trained LDA model from which to extract topics.
    corpus : list
        A list of documents represented as bag-of-words. Each document is a list of (word_id, word_frequency) tuples.
    num_topics : int, optional
        The number of topics to retrieve for each document. Default is 10.

    Returns:
    -------
    list of str
        A list where each element is a formatted string representing the dominant topics and their probabilities for each document.

    Examples:
    --------
    >>> lda_model = train_lda_model(corpus, dictionary, num_topics=5)
    >>> topics = extract_topics(lda_model, corpus)
    >>> print(topics[0])  # Outputs formatted topics for the first document in the corpus.
    """
    top_topics_per_document = [
        lda_model.get_document_topics(item, minimum_probability=0) for item in corpus
    ]

    formatted_topics_str = [
        "; ".join(
            [f"Topic {topic_num}: {prob:.2f}" for topic_num, prob in doc if prob > 0.01]
        )
        for doc in top_topics_per_document
    ]
    return formatted_topics_str


def enrich_dataframe(
    topic_df: pd.DataFrame, topics: List[str], column_name: str = "top_topics"
) -> pd.DataFrame:
    """
    Adds a new column to the provided DataFrame with extracted topics for each document.

    Parameters:
    ----------
    topic_df : pandas.DataFrame
        DataFrame containing the data where the new column will be added. This DataFrame should have a structure compatible with the topics being appended.
    topics : list of str
        A list of strings where each string contains formatted topics extracted from each document.
    column_name : str, optional
        The name of the new column to be added to the DataFrame which will contain the topics. Default is 'top_topics'.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with an additional column containing the formatted topics for each document.

    Examples:
    --------
    >>> topic_df = pd.DataFrame({'email_content': ['text about health', 'text about finance']})
    >>> topics = ['Topic 1: 0.70; Topic 2: 0.30', 'Topic 1: 0.50; Topic 2: 0.50']
    >>> enriched_df = enrich_dataframe(topic_df, topics)
    >>> print(enriched_df.head())
    """
    topic_df = topic_df.copy()
    topic_df[column_name] = topics
    return topic_df
