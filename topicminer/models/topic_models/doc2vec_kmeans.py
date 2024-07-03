import pandas as pd
import numpy as np
from IPython.display import display
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from topicminer.config import PROCESSED_TEXT_COL

class Doc2Vec_KMeans_Cat:
    def __init__(self, vector_size=100, window=5, min_count=5, workers=16, epochs=10, n_clusters=10, num_unique_terms=10):
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
        self.documents = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL])[PROCESSED_TEXT_COL].to_list()
        self.tagged_documents = [TaggedDocument(words=doc.split(), tags=[i]) for i, doc in enumerate(self.documents)]

    def train_doc2vec(self):
        self.model = Doc2Vec(self.tagged_documents, vector_size=self.vector_size, window=self.window,
                             min_count=self.min_count, workers=self.workers, epochs=self.epochs)

    def categorize_documents(self):
        # Train Doc2Vec if not already trained
        if self.model is None:
            self.train_doc2vec()

        # Create document vectors
        doc_vectors = np.array([self.model.dv[i] for i in range(len(self.tagged_documents))])
    
        # Cluster document vectors
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(doc_vectors)
    
        # Extract top terms
        key_terms = self.extract_key_terms(doc_vectors)
    
        # Organize data into a DataFrame for better manipulation and display
        emails_df = pd.DataFrame({
            'Email': self.documents,
            'Cluster': self.kmeans.labels_
        })
        
        return emails_df, key_terms

    def extract_key_terms(self, doc_vectors):
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

    def show_clusters_with_emails(self):
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
        emails_df = pd.merge(emails_df, key_terms, left_on='Cluster', right_index=True)
        emails_df.rename(columns={0:'Key Terms'}, inplace=True)
        return emails_df