import pandas as pd
import numpy as np

from IPython.display import display

from topicminer.config import PROCESSED_TEXT_COL

class Word2Vec_KMeans_Cat:
    def __init__(self, vector_size=100, window=5, min_count=5, workers=8, epochs=10, n_clusters=10, num_unique_terms=10):
        from topicminer import read_json_to_dataframe
        if workers > 8:
            print("Workers can be no higher than 8. Resetting to 8.")
            workers = 8
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
        self.tokenized_docs = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL])[PROCESSED_TEXT_COL].to_list()

    def train_word2vec(self):
        from gensim.models import Word2Vec
        self.model = Word2Vec(self.tokenized_docs, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, workers=self.workers, epochs=self.epochs)

    def document_vector(self, doc):                     
        """Averaging word vectors in a document."""
        words = [word for word in doc if word in self.model.wv.index_to_key]
        if words:
            return np.mean(self.model.wv[words], axis=0)
        else:
            return np.zeros(self.vector_size)

    def categorize_documents(self):
        # Train Word2Vec if not already trained
        if self.model is None:
           self.train_word2vec()

        # Create document vectors
        doc_vectors = np.array([self.document_vector(doc) for doc in self.tokenized_docs])
    
        # Cluster document vectors
        # Ensure to return also the emails or their identifiers along with the clusters
        from sklearn.cluster import KMeans
        doc_vectors = np.array([self.document_vector(doc) for doc in self.tokenized_docs])
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(doc_vectors)
    
        # Extract top terms
        key_terms = self.extract_key_terms()
    
        # Map cluster labels to key terms
        cluster_labels = self.kmeans.labels_
        
        # Organize data into a DataFrame for better manipulation and display
        emails_df = pd.DataFrame({
            'Email': self.tokenized_docs,
            'Cluster': cluster_labels
        })
        
        
        key_terms = self.extract_key_terms()

        return emails_df, key_terms

    def extract_key_terms(self):
        from sklearn.metrics.pairwise import cosine_similarity
        centroids = self.kmeans.cluster_centers_
        terms = {}
        for i in range(self.n_clusters):
            centroid = centroids[i]
            cos_similarities = cosine_similarity([centroid], np.array([self.document_vector(doc) for doc in self.tokenized_docs]))
            top_docs_indices = np.argsort(cos_similarities[0])[-3:]  # Get indices of the top 3 closest documents
            cluster_terms = []
            for idx in top_docs_indices:
                cluster_terms.extend(self.tokenized_docs[idx].split())  # Assuming tokenized_docs are properly split into words
            unique_terms = list(set(cluster_terms))
            terms[i] = unique_terms[:self.num_unique_terms] if len(unique_terms) >= self.num_unique_terms else unique_terms
        return terms
    
    def show_topics(self):
        emails_df, key_terms = self.categorize_documents()
        labels_df = pd.DataFrame.from_dict(key_terms, orient='index')
        labels_df.index.name = 'Labels'
        labels_df.columns = [f'Word[{i}]' for i in range(labels_df.shape[1])]
        return labels_df

    def show_emails_with_topics(self):
        emails_df, key_terms = self.categorize_documents()
        
        # Ensure the DataFrame is correctly formed
        summary_df = pd.DataFrame.from_dict(key_terms, orient='index')
        # Adjusting columns dynamically based on the actual number of key terms per cluster
        summary_df.columns = [f'Key Term {i}' for i in range(summary_df.shape[1])]
        summary_df.index.name = 'Cluster'
        
        # Display summary DataFrame
        print("Summary of Clusters and Key Terms:")
        display(summary_df)
        
        # Display emails grouped by Cluster
        for cluster in sorted(emails_df['Cluster'].unique()):
            print(f"\nCluster {cluster} Emails:")
            display(emails_df[emails_df['Cluster'] == cluster][['Email']])

