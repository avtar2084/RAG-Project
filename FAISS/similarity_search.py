# !pip install faiss-cpu numpy scikit-learn
# !pip install "tensorflow>=2.0.0"
# !pip install --upgrade tensorflow-hub


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# Suppressing warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

pprint(list(newsgroups_train.target_names))

# Display the first 3 posts from the dataset
for i in range(3):
    print(f"Sample post {i+1}:\n")
    pprint(newsgroups_train.data[i])
    print("\n" + "-"*80 + "\n")

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Basic preprocessing of text data
def preprocess_text(text):
    # Remove email headers
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess each document
processed_documents = [preprocess_text(doc) for doc in documents]

# Choose a sample post to display
sample_index = 0  # for example, the first post in the dataset

# Print the original post
print("Original post:\n")
print(newsgroups_train.data[sample_index])
print("\n" + "-"*80 + "\n")

# Print the preprocessed post
print("Preprocessed post:\n")
print(preprocess_text(newsgroups_train.data[sample_index]))
print("\n" + "-"*80 + "\n")

# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to generate embeddings
def embed_text(text):
    return embed(text).numpy()

# Generate embeddings for each preprocessed document
X_use = np.vstack([embed_text([doc]) for doc in processed_documents])

dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)  # Creating a FAISS index
index.add(X_use)  # Adding the document vectors to the index

# Function to perform a query using the Faiss index
def search(query_text, k=5):
    # Preprocess the query text
    preprocessed_query = preprocess_text(query_text)
    # Generate the query vector
    query_vector = embed_text([preprocessed_query])
    # Perform the search
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

# Example Query
query_text = "motorcycle"
distances, indices = search(query_text)

# Display the results
for i, idx in enumerate(indices[0]):
    # Ensure that the displayed document is the preprocessed one
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{processed_documents[idx]}\n")


# Display the results
for i, idx in enumerate(indices[0]):
    # Displaying the original (unprocessed) document corresponding to the search result
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{documents[idx]}\n")

