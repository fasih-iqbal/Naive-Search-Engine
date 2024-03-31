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


def calculate_tf_idf_query(query, corpus, df_array, num_documents):
    """
    Calculates TF-IDF score for each word in the query based on the corpus document frequency.
    """
    preprocessed_query = preprocess_text(query)
    tf_query = {word: 0 for word in corpus}
    for word in preprocessed_query:
        if word in corpus:
            tf_query[word] += 1

    tf_idf_query = {}
    for word, tf in tf_query.items():
        if tf > 0 and word in corpus:
            word_index = corpus[word]
            idf = np.log(num_documents / (1 + df_array[word_index]))
            tf_idf_query[word_index] = tf * idf

    return tf_idf_query


def calculate_cosine_similarity(doc_vector, query_vector):
    """
    Calculates the cosine similarity between the document vector and the query vector.
    """
    dot_product = sum([doc_vector[word_index] * query_vector.get(word_index, 0)
                       for word_index in doc_vector])
    doc_vector_magnitude = math.sqrt(
        sum([val**2 for val in doc_vector.values()]))
    query_vector_magnitude = math.sqrt(
        sum([val**2 for val in query_vector.values()]))

    if doc_vector_magnitude == 0 or query_vector_magnitude == 0:
        return 0
    else:
        cosine_similarity = dot_product / \
            (doc_vector_magnitude * query_vector_magnitude)
        return cosine_similarity


# Main mapper function
for line in sys.stdin:
    # Split the input into document ID and text
    doc_id, text = line.strip().split('\t')
    # Preprocess the text
    preprocessed_words = preprocess_text(text)
    # Calculate TF for each word in the document
    tf_article = {}
    for word in preprocessed_words:
        if word not in corpus:
            corpus[word] = len(corpus)
        if corpus[word] not in tf_article:
            tf_article[corpus[word]] = 1
        else:
            tf_article[corpus[word]] += 1
    # Emit (word_index, doc_id:tf) pairs
    for word_index, tf in tf_article.items():
        print(f"{word_index}\t{doc_id}:{tf}")

# Print corpus for debugging
# print("Corpus:")
# for word, index in corpus.items():
#     print(f"{word}: {index}")
