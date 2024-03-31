import sys
import math
import numpy as np
from collections import OrderedDict

# Initialize stopwords and corpus
stop_words = set([
    # List of stopwords
])

corpus = OrderedDict()


def preprocess_text(text):
    """
    Preprocesses the text by removing stopwords, punctuation,
    and converting it to lowercase.
    """
    # Remove punctuation
    text = ''.join([char.lower()
                   for char in text if char.isalnum() or char.isspace()])
    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


# Initialize variables
num_documents = 0
df_array = np.zeros(len(corpus))
weights = {}

# Read input from mapper
for line in sys.stdin:
    word_index, doc_tf = line.strip().split('\t')
    doc_id, tf = doc_tf.split(':')
    word_index = int(word_index)
    tf = int(tf)

    # Increment document frequency count
    df_array[word_index] += 1

    # Update weights dictionary
    if doc_id not in weights:
        weights[doc_id] = {}
    weights[doc_id][word_index] = tf

    # Update corpus if needed
    if word_index not in corpus:
        corpus[word_index] = len(corpus)

    # Update number of documents
    num_documents += 1

# Calculate IDF for each word
idf_array = np.log(num_documents / (1 + df_array))

# Calculate TF/IDF weights
for doc_id, doc_weights in weights.items():
    tfidf_weights = {}
    for word_index, tf in doc_weights.items():
        tfidf_weights[word_index] = tf * idf_array[word_index]
    weights[doc_id] = tfidf_weights

# Process query


def process_query(query, corpus, df_array, num_documents):
    preprocessed_query = preprocess_text(query)
    tf_query = {}
    for word in preprocessed_query:
        if word in corpus:
            word_index = corpus[word]
            tf_query[word_index] = tf_query.get(word_index, 0) + 1

    # Calculate TF/IDF for query
    tfidf_query = {}
    for word_index, tf in tf_query.items():
        tfidf_query[word_index] = tf * idf_array[word_index]

    return tfidf_query

# Calculate cosine similarity and rank documents


def rank_documents(query, corpus, df_array, num_documents, weights):
    tfidf_query = process_query(query, corpus, df_array, num_documents)
    relevance_scores = {}
    for doc_id, doc_weights in weights.items():
        dot_product = sum(doc_weights.get(
            word_index, 0) * tfidf_query.get(word_index, 0) for word_index in doc_weights)
        doc_magnitude = math.sqrt(sum(val**2 for val in doc_weights.values()))
        query_magnitude = math.sqrt(
            sum(val**2 for val in tfidf_query.values()))
        if doc_magnitude == 0 or query_magnitude == 0:
            relevance_scores[doc_id] = 0
        else:
            relevance_scores[doc_id] = dot_product / \
                (doc_magnitude * query_magnitude)
    return relevance_scores

# Print ranked documents


def print_ranked_documents(relevance_scores, output_length):
    print("\nRanked Documents (by relevance):")
    for doc_id, score in sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)[:output_length]:
        print(f"Document {doc_id}: {score:.2f}")
