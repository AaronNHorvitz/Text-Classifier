# topicminer/models/topic_modles/doc2vec_kmeans.py

"""
This module implements the Doc2Vec_KMeans_Cat class which combines the use of Doc2Vec for generating document embeddings
and KMeans clustering to categorize documents into meaningful clusters. It provides functionality for training the Doc2Vec model,
clustering documents based on their embeddings, extracting key terms from clusters, and displaying organized summaries of the clusters.

Classes
-------
Doc2Vec_KMeans_Cat

Usage
-----
The class is designed to be used in data analysis workflows where text data clustering and categorization are required, particularly
for organizing large collections of textual data into manageable and interpretable groups based on content similarity.
"""

# Standard library imports
import numpy as np
import pandas as pd

# Related third-party imports
from IPython.display import display
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Local application/library specific imports
from topicminer.config import PROCESSED_TEXT_COL


class Doc2Vec_KMeans_Cat:
    """
    A class to encapsulate the Doc2Vec and KMeans clustering functionality for
    categorizing documents based on their semantic content.

    This class first converts text documents into vector representations using
    the Doc2Vec model, which learns to represent documents in a continuous vector space.
    It then applies KMeans clustering to categorize these documents into a specified
    number of clusters based on their vector similarities.

    Attributes:
        vector_size (int): Dimensionality of the feature vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with total frequency lower than this.
        workers (int): Number of worker threads to train the model (faster training with multicore machines).
        epochs (int): Number of iterations (epochs) over the corpus.
        n_clusters (int): Number of clusters to form.
        num_unique_terms (int): Number of unique terms to retrieve per cluster for summary.
        model (Doc2Vec): The trained Doc2Vec model.
        kmeans (KMeans): The KMeans clustering model.
        documents (list of str): Loaded documents for processing.
        tagged_documents (list of TaggedDocument): Documents tagged with identifiers for Doc2Vec training.

    Methods:
        train_doc2vec(): Trains the Doc2Vec model using the tagged documents.
        categorize_documents(): Categorizes documents by clustering their vector representations.
        extract_key_terms(doc_vectors): Extracts key terms for each cluster.
        show_topics(): Displays the clusters and their key terms in a dataframe.
        group_emails_by_cluster(): Groups emails by their cluster and displays them.
        show_clusters_with_emails(): Links clusters with corresponding emails for display.
    """

    def __init__(
        self,
        vector_size=100,
        window=5,
        min_count=5,
        workers=16,
        epochs=10,
        n_clusters=10,
        num_unique_terms=10,
    ):
        from topicminer import read_json_to_dataframe

        if workers > 16:
            print("Workers can be no higher than 16. Resetting to 16.")
            workers = 16
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.n_clusters = n_clusters
        self.num_unique_terms = num_unique_terms
        self.model = None
        self.kmeans = None

        # Load and store tokenized documents at initialization
        self.documents = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL])[
            PROCESSED_TEXT_COL
        ].to_list()
        self.tagged_documents = [
            TaggedDocument(words=doc.split(), tags=[i])
            for i, doc in enumerate(self.documents)
        ]

    def train_doc2vec(self):
        """
        Trains the Doc2Vec model on the tagged documents.
        """
        self.model = Doc2Vec(
            self.tagged_documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
        )

    def categorize_documents(self):
        """
        Uses Doc2Vec vectors to cluster documents and returns their groupings along with key terms.
        """
        # Train Doc2Vec if not already trained
        if self.model is None:
            self.train_doc2vec()

        # Create document vectors
        doc_vectors = np.array(
            [self.model.dv[i] for i in range(len(self.tagged_documents))]
        )

        # Cluster document vectors
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(
            doc_vectors
        )

        # Extract top terms
        key_terms = self.extract_key_terms(doc_vectors)

        # Organize data into a DataFrame for better manipulation and display
        emails_df = pd.DataFrame(
            {"Email": self.documents, "Cluster": self.kmeans.labels_}
        )

        return emails_df, key_terms

    def extract_key_terms(self, doc_vectors):
        """
        Identifies key terms in each cluster by analyzing the top documents closest to the cluster's centroid.
        """
        centroids = self.kmeans.cluster_centers_
        terms = {}
        for i in range(self.n_clusters):
            centroid = centroids[i]
            cos_similarities = cosine_similarity([centroid], doc_vectors)
            top_docs_indices = np.argsort(cos_similarities[0])[-3:]
            cluster_terms = []
            for idx in top_docs_indices:
                cluster_terms.extend(self.documents[idx].split())
            unique_terms = list(set(cluster_terms))
            terms[i] = (
                unique_terms[: self.num_unique_terms]
                if len(unique_terms) >= self.num_unique_terms
                else unique_terms
            )
        return terms

    def show_topics(self):
        """
        Displays a summary DataFrame showing clusters and their associated key terms.
        """
        emails_df, key_terms = self.categorize_documents()
        labels_df = pd.DataFrame.from_dict(key_terms, orient="index")
        labels_df.index.name = "Cluster"
        labels_df.columns = [f"Key Term[{i}]" for i in range(labels_df.shape[1])]
        return labels_df

    def group_emails_by_cluster(self):
        """
        Groups and displays emails by their assigned cluster, showing the cluster summary.
        """
        emails_df, key_terms = self.categorize_documents()

        # Ensure the DataFrame is correctly formed
        summary_df = pd.DataFrame.from_dict(key_terms, orient="index")
        # Adjusting columns dynamically based on the actual number of key terms per cluster
        summary_df.columns = [f"Key Term {i}" for i in range(summary_df.shape[1])]
        summary_df.index.name = "Cluster"

        # Display summary DataFrame
        print("Summary of Clusters and Key Terms:")
        display(summary_df)

        # Display emails grouped by Cluster
        for cluster in sorted(emails_df["Cluster"].unique()):
            print(f"\nCluster {cluster} Emails:")
            display(emails_df[emails_df["Cluster"] == cluster][["Email"]])
            print("\n")

    def show_clusters_with_emails(self):
        """
        Merges and displays emails with their clusters and key terms, showing the detailed email content.
        """
        from topicminer import read_json_to_dataframe

        emails_df = read_json_to_dataframe()[
            ["email_subject", "email_body", "email_categories"]
        ]
        clusters = self.categorize_documents()[0]["Cluster"]
        emails_df["Cluster"] = clusters

        # Obtain key terms and add them to the dataframe
        tokens_df, key_terms = self.categorize_documents()

        # Transforms the key terms from individual terms in colums to rows of lists and feed them back into the dataframe.
        key_terms = pd.DataFrame(key_terms)
        dict = {}
        for i, row in key_terms.iterrows():
            row_ = row.to_list()
            row_str = ", ".join(row_)
            row_ = [row_str]
            dict[i] = row_
        # Put key terms into a dataframe.
        key_terms = pd.DataFrame(dict).T

        # Add key terms to the emails dataframe.
        emails_df = pd.merge(emails_df, key_terms, left_on="Cluster", right_index=True)
        emails_df.rename(columns={0: "Key Terms"}, inplace=True)
        return emails_df
