# text_indexer_and_search.py
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from gensim.models import Word2Vec
import faiss

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Preprocessing function
def preprocess(document):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(document.lower())
    filtered_words = [word for word in tokens if word not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)

# Process documents
processed_documents = [preprocess(doc) for doc in documents]

# Build TF-IDF model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_documents)

# Save the inverted index
with open('tfidf_inverted_index.pkl', 'wb') as f:
    pickle.dump((vectorizer, tfidf_matrix), f)

# Function to search documents
def search(query: str):
    with open('tfidf_inverted_index.pkl', 'rb') as f:
        vectorizer, tfidf_matrix = pickle.load(f)
    query_processed = preprocess(query)
    query_tfidf = vectorizer.transform([query_processed])
    cos_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_docs = cos_similarities.argsort()[-3:][::-1]  # Top 3 documents
    return top_docs, cos_similarities[top_docs]

# Optional: Load or create Word2Vec model and use FAISS for efficient searching
def create_embedding_model(processed_docs):
    # Load a pre-trained model or train one
    # model = api.load('word2vec-google-news-300')
    model = Word2Vec(sentences=[doc.split() for doc in processed_docs], vector_size=100, window=5, min_count=1, workers=4)
    doc_vectors = np.array([np.mean([model.wv[word] for word in words.split() if word in model.wv]
                                    or [np.zeros(model.vector_size)], axis=0)
                            for words in processed_documents])
    return doc_vectors

# Create FAISS index
def create_faiss_index(doc_vectors):
    index = faiss.IndexFlatL2(doc_vectors.shape[1])  # L2 distance for similarity
    index.add(doc_vectors.astype(np.float32))  # Add vectors to the index
    return index

# Usage
if __name__ == '__main__':
    # Search
    top_docs, similarities = search("first document")
    print("Top matching documents:", top_docs)
    print("Similarities:", similarities)

    # Optional: Vector embeddings with FAISS
    doc_vectors = create_embedding_model(processed_documents)
    index = create_faiss_index(doc_vectors)
    # Search with FAISS (query vector must be from the same embedding model)
    query_vector = doc_vectors[0]  # We want to find similar to first doc
    k = 3  # Number of nearest neighbors
    D, I = index.search(np.array([query_vector]).astype(np.float32), k)  # Distance and indices
    print("FAISS search results indices:", I)
    print("FAISS search distances:", D)
