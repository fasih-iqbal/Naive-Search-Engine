import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import OrderedDict
import numpy as np

# Load your subset dataframe
file_path = r"C:\Users\PC\Documents\Symmester 4\Big Data\Assignment 2\subset.csv"
subset_df = pd.read_csv(file_path)

# Function to preprocess text


def preprocess_text(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [word.lower() for word in words if word.isalnum()
                          and word.lower() not in stop_words]
        return filtered_words
    else:
        return []


# Create a corpus
corpus = OrderedDict()

# Iterate through the rows to build the corpus
for index, row in subset_df.iterrows():
    article_text = row['SECTION_TEXT']
    preprocessed_words = preprocess_text(article_text)
    for word in preprocessed_words:
        if word not in corpus:
            corpus[word] = len(corpus)

# Initialize a dictionary to store TF for each article
tf_dict = {}

# Iterate through the rows to calculate TF
for index, row in subset_df.iterrows():
    article_text = row['SECTION_TEXT']
    preprocessed_words = preprocess_text(article_text)
    article_tf = {idx: 0 for idx in corpus.values()}
    for word in preprocessed_words:
        if word in corpus:
            article_tf[corpus[word]] += 1
    article_tf = {idx: freq for idx, freq in article_tf.items() if freq > 0}
    tf_dict[index] = article_tf

# Initialize an array to store the document frequency (DF) for each word
df_array = np.zeros(len(corpus))

# Iterate through the TF dictionary to calculate DF
for tf in tf_dict.values():
    for word_idx in tf.keys():
        df_array[word_idx] += 1

# Initialize a 2D array to store document vectors
document_vectors = []

# Iterate through the TF dictionary to create vectors for each document
for tfidf in tf_dict.values():
    document_vector = [0] * len(corpus)
    for word_idx, tfidf_value in tfidf.items():
        document_vector[word_idx] = tfidf_value
    document_vectors.append(document_vector)

# Function to calculate relevance between query vector and document vector


def calculate_relevance(query_vector, document_vector):
    relevance = 0
    for term_idx, tfidf_query in enumerate(query_vector):
        if tfidf_query != 0:
            relevance += tfidf_query * document_vector[term_idx]
    return relevance


# Get user input for the query text
query_text = input("Enter your query text: ")

# Preprocess the query text
preprocessed_query = preprocess_text(query_text)

# Initialize a dictionary to store TF for the query
query_tf = {idx: 0 for idx in corpus.values()}

# Calculate TF for the query
for word in preprocessed_query:
    if word in corpus:
        query_tf[corpus[word]] += 1

# Create the query vector
query_vector = [0] * len(corpus)
for word_idx, tf_value in query_tf.items():
    query_vector[word_idx] = tf_value

# Initialize a list to store relevance scores along with document index
document_relevance = []

# Calculate relevance between query vector and each document vector
for i, doc_vector in enumerate(document_vectors):
    relevance = calculate_relevance(query_vector, doc_vector)
    document_relevance.append((i, relevance))

# Sort the document relevance list by relevance score
document_relevance.sort(key=lambda x: x[1], reverse=True)

# Display the top 10 relevant documents
for i, (doc_index, relevance) in enumerate(document_relevance[:10], start=1):
    print(f"Relevance between query and document {doc_index + 1}: {relevance}")
