import pandas as pd
import numpy as np
import json
from IPython.display import display
import logging
from tqdm import tqdm


from topicminer.config import PROCESSED_TEXT_COL, PROCESSED_EMAILS_JSON_FILE 

class Word2Vec_KMeans_Cat:
    def __init__(self, vector_size=100, window=5, min_count=5, workers=16, epochs=10, n_clusters=10, num_unique_terms=10):
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
        labels_df.index.name = 'Cluster'
        labels_df.columns = [f'Key Term[{i}]' for i in range(labels_df.shape[1])]
        return labels_df

    def group_emails_by_cluster(self):
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
            print("\n")

    def show_clusters_with_emails(self, num_emails=10):
        from topicminer import read_json_to_dataframe

        emails_df = read_json_to_dataframe()[['email_subject','email_body','email_categories']]
        clusters = self.categorize_documents()[0]['Cluster']
        emails_df['Cluster'] = clusters

        # Obtain key terms and add them to the dataframe
        tokens_df, key_terms = self.categorize_documents()

        # Transforms the key terms from individual terms in colums to rows of lists and feed them back into the dataframe.
        key_terms = pd.DataFrame(key_terms)
        dict = {}
        for i,row in key_terms.iterrows():
            row_ = row.to_list()
            row_str = ', '.join(row_)
            row_ = [row_str]
            dict[i] = row_
        # Put key terms into a dataframe. 
        key_terms = pd.DataFrame(dict).T

            # Add key terms to the emails dataframe.

    def document_vector(self, doc):
        """Averaging word vectors in a document."""
        words = [word for word in doc if word in self.model.wv.index_to_key]
        if words:
            # Ensure the vector is of type float64
            return np.mean(self.model.wv[words], axis=0).astype(np.float64)
        else:
            return np.zeros(self.vector_size, dtype=np.float64)

    def embed_cluster_data(self):
        # Ensure the model and clustering are trained
        if self.model is None or self.kmeans is None:
            _, _ = self.categorize_documents()  # This ensures both Word2Vec and KMeans are trained

        # Load the existing data
        with open(PROCESSED_EMAILS_JSON_FILE, 'r') as file:
            data = json.load(file)

        # Process each document
        for doc in data:
            doc_vector = self.document_vector(doc['processed_text'].split())
            doc_vector = np.array(doc_vector, dtype=np.float64).reshape(1, -1)  # Ensure correct data type for KMeans
            cluster_label = self.kmeans.predict(doc_vector)[0]  # Predict the cluster label for this document
            doc['cluster'] = int(cluster_label)  # Convert numpy int to Python int
            # Extract key terms and convert numpy types if needed
            key_terms = [str(term) for term in self.extract_key_terms()[cluster_label]]
            doc['key_terms'] = ', '.join(key_terms)

            # Calculate and assign cluster probabilities if needed
            distances = self.kmeans.transform(doc_vector).flatten()
            probabilities = np.exp(-distances)  # Convert distances to a probability scale
            probabilities /= probabilities.sum()  # Normalize to sum to 1
            # Create a dictionary of cluster probabilities
            doc['cluster_probabilities'] = {str(i): float(prob) for i, prob in enumerate(probabilities)}

        # Save the updated data back to the JSON file
        with open(PROCESSED_EMAILS_JSON_FILE, 'w') as file:
            json.dump(data, file, indent=4)

        print("Cluster data and key terms embedded into JSON file successfully.")