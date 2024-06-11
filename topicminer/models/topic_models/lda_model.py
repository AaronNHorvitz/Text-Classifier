import pandas as pd
import numpy as np

# Machine Learning and topic modeling
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LdaMulticore, TfidfModel, Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def train_lda_model(
    corpus: list,
    gensim_dictionary: Dictionary,
    num_topics: int = 10,
    workers: int = 8,
    chunksize: int = 2000,
    passes=1,
    batch=False,
    alpha: str = "auto",
    eta=None,
    decay=0.5,
    offset=1,
    eval_every=10,
    iterations=50,
    gamma_threshold=0.001,
    random_state: int = 100,
    minimum_probability=0.01,
    minimum_phi_value=0.01,
    per_word_topics=False,
    dtype=np.float32
) -> LdaMulticore:
    """
    Trains an LDA (Latent Dirichlet Allocation) model using the Gensim library with support for multicore processing.
    This function sets up and runs the LDA model training using the provided corpus and dictionary. It can be configured
    with a number of parameters to optimize the training according to specific needs.

    Parameters:
    ----------
    corpus : list
        A list of bag-of-words (word_id, word_frequency) tuples representing the corpus.
    gensim_dictionary : Dictionary
        A Gensim Dictionary object mapping IDs to words, which will be used during training to map word IDs to words.
    num_topics : int, optional
        The number of latent topics to extract from the corpus. Default is 10.
    workers : int, optional
        The number of worker processes to use for parallelization. If more than 8, it will be set to 8. Default is 8.
    chunksize : int, optional
        The number of documents to process at a time in the training algorithm. Default is 2000.
    passes : int, optional
        The number of passes over the corpus during training. Default is 1.
    batch : bool, optional
        Whether to use all documents in each training chunk. Default is False.
    alpha : str, optional
        The hyperparameter affecting the sparsity/thickness of the topics. Default is 'auto'.
    eta : optional
        The eta hyperparameter, affects topic-word density. The default is None, which sets it to 1/num_topics.
    decay : float, optional
        Affects learning rate over time. Default is 0.5.
    offset : float, optional
        Hyperparameter that downweights early iterations. Default is 1.
    eval_every : int, optional
        Determines how often the model parameters should be updated. Default is 10.
    iterations : int, optional
        Maximum number of iterations over each document. Default is 50.
    gamma_threshold : float, optional
        Convergence threshold for gamma updates. Default is 0.001.
    random_state : int, optional
        Random state for reproducibility. Default is 100.
    minimum_probability : float, optional
        Topics with a probability lower than this threshold will be filtered out. Default is 0.01.
    minimum_phi_value : float, optional
        Topics with a phi value lower than this threshold will be filtered out. Default is 0.01.
    per_word_topics : bool, optional
        If True, the model also computes a list of topics for each word. Default is False.
    dtype : data-type, optional
        The data type to use in the computation (default is np.float32).

    Returns:
    -------
    LdaMulticore
        The trained LDA model.

    Examples:
    --------
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import LdaMulticore
    >>> dictionary = Dictionary(documents)
    >>> corpus = [dictionary.doc2bow(text) for text in documents]
    >>> lda_model = train_lda_model(corpus, dictionary, num_topics=5, passes=15, workers=4)
    """
    if workers > 8:
        workers = 8
        print("The maximum number of workers allowed is 8. Setting workers to 8.")

    lda_model = LdaMulticore(
        corpus=corpus,
        num_topics=num_topics,
        id2word=gensim_dictionary,
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
        dtype=dtype
        )

    # tqdm progress bar for the remaining passes, if more than one pass is required.
    if passes > 1:
        for pass_idx in tqdm(range(1, passes), desc="Training LDA Model"):
            lda_model.update(corpus)  # Update the model in subsequent passes.

    return lda_model


def perform_lda_topic_modelling(
    emails_df: pd.DataFrame,
    processed_text_col: str = "processed_text",
    num_topics: int = 10,
    random_state: int = 100,
    chunksize: int = 2000,
    passes: int = 10,
    alpha: str = "symmetric",
    workers: int = 8,
    no_below: int = 5,  # Default setting for minimum document frequency
    no_above: float = 0.5,  # Default setting for maximum document proportion
) -> Tuple[pd.DataFrame, LdaModel]:
    """
    Performs Latent Dirichlet Allocation (LDA) topic modeling on a collection of emails to identify prevalent topics
    within the corpus. This function enriches the input DataFrame by appending a column that lists the most significant
    topics in each document, based on their contribution weights.

    Parameters:
    ----------
    emails_df : pandas.DataFrame
        DataFrame containing the emails, with at least one column of preprocessed text.
    processed_text_col : str, default 'processed_text'
        Name of the column in `emails_df` that contains preprocessed text for topic modeling.
    num_topics : int, default 10
        The number of distinct topics to identify in the LDA model.
    random_state : int, default 100
        Seed for the random number generator for reproducibility.
    chunksize : int, default 2000
        Number of documents to consider at once in the training algorithm.
    passes : int, default 10
        Number of training passes through the corpus.
    alpha : {'symmetric', 'asymmetric', 'auto'} or list, default 'symmetric'
        Hyperparameter affecting document-topic density. Can be specified as a list for asymmetric priors.
    workers : int, default 8
        Number of worker processes to use for parallelization. If None, uses all available cores minus one.
    no_below : int, default 5
        Minimum number of documents a token must appear in to be kept. This helps to remove rare words which may have
        less significance in topic modeling.
    no_above : float, default 0.5
        Maximum proportion of documents a token can appear in to be kept. This helps to remove too common words which
        may dominate the topic but have little informative value.

    Returns:
    -------
    tuple
        A tuple containing:
        - DataFrame: The input DataFrame enriched with a new column 'top_topics' that lists the dominant topics and their
          weights for each document.
        - model: The trained LdaModel, which can be used for further analysis or visualization of topics.

    Examples:
    --------
    >>> emails_df = pd.DataFrame({
            'processed_text': ["text about health", "text about finance"]
        })
    >>> enriched_df, lda_model = perform_lda_topic_modelling(emails_df)
    >>> print(enriched_df['top_topics'].head())

    Notes:
    -----
    This function is crucial for understanding large collections of text by breaking them down into manageable themes
    or topics. It's particularly useful in exploratory data analysis and natural language processing applications where
    the themes of documents need to be understood quickly.
    """

    # Create dictionary and corpus using the new unified function
    dictionary, corpus = prepare_corpus_and_dictionary(
        emails_df, processed_text_col, no_below, no_above
    )

    # Apply TF-IDF transformation
    corpus_tfidf = TfidfModel(corpus)[corpus]

    # Train the LDA model
    lda_model_tfidf = train_lda_model(
        corpus_tfidf,
        dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=chunksize,
        passes=passes,
        alpha=alpha,
        workers=workers,
    )

    # Extract top topics for each document
    top_topics_per_document = [
        lda_model_tfidf.get_document_topics(item) for item in corpus_tfidf
    ]
    top_topics_str = [
        "; ".join([f"Topic {topic_num}: {prob:.2f}" for topic_num, prob in doc])
        for doc in top_topics_per_document
    ]

    # Enrich DataFrame with top topics
    enriched_emails_df = emails_df.copy()
    enriched_emails_df["top_topics"] = top_topics_str

    return enriched_emails_df, lda_model_tfidf

