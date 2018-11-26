###
# A Q&A system converting the question into embedding
# And finding the answer from the persistance storage
# Consisting embeddings and sentences.
###

# Below line can be used in a Jupyter notebook to enable intellisense (TAB after '.')
# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import unicodedata

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# From persistance storage to in-memory for comparison
def embeddings_file_to_list(pathToFile):
    file = open(pathToFile, 'r')
    text = file.read()
    data = unicodedata.normalize("NFKD", text)
    list_of_vectors = []
    string_vector_list = data.split("\n")
    string_vector_list.pop()
    for i, sent in enumerate(np.array(string_vector_list).tolist()):
        vector = [float(s) for s in sent.split(',')]
        list_of_vectors.append(vector)
    list_of_vectors = np.array(list_of_vectors).astype(np.float32)
    file.close()
    return list_of_vectors

preSavedEmbeddings = embeddings_file_to_list('embeddingspb.txt')

print(len(preSavedEmbeddings))
print(len(preSavedEmbeddings[0]))

print(preSavedEmbeddings[:5])

# Your question goes here
question = ['']

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    question_embeddings = session.run(embed(question))

# Returning index and similarity of the most similar embedding
def most_similar_index(ques_emb, sent_emb):
    index = -1
    similarity = 0
    
    for i, sent in enumerate(np.array(sent_emb).tolist()):
        score = np.inner(sent, ques_emb)
        if score > similarity:
            similarity = score
            index = i
    
    return index, similarity

index, similarity = most_similar_index(question_embeddings, preSavedEmbeddings)

# Retrieving sentence using index
path = ""
raw = open(path, 'r')
text = raw.read()
data = unicodedata.normalize("NFKD", text)
raw.close()
data = data.replace('\n', ' ')
data = data.replace('\r', ' ')
data = data.replace('\t', ' ')
data = data.replace('. ', '.')
sentences = data.split(".")

print('Answer: ', sentences[index])
print('Similarity score: ', similarity)