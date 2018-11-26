###
# Converting each sentence of a text file into embeddings (vector representation)
# And storing them in another file
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

# Reading the text
filename = ''
path = "" + filename
raw = open(path, 'r')
text = raw.read()
data = unicodedata.normalize("NFKD", text)
raw.close()
data = data.replace('\n', '')
data = data.replace('\r', '')
data = data.replace('\t', '')
data = data.replace('. ', '.')
sentences = data.split(".")
sentences[:10]

# In-memory embeddings storage
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    sentence_embeddings = session.run(embed(sentences))

print("No. of sentences: ", len(sentences))
print("No. of embeddings: ", len(sentence_embeddings))
print("Dimension of embedding (vector): ", len(sentence_embeddings[0]))

print("Embeddings: \n", sentence_embeddings[:5])

# Persisting in-memory store
def embeddings_list_to_file(embeddings, filename):
    with open('embeddings_'+filename, 'a') as file:
        for i, emb in enumerate(np.array(embeddings).tolist()):
            file.write(','.join(str(e) for e in emb))
            file.write('\n')
        file.close()

embeddings_list_to_file(sentence_embeddings, filename)