import pickle
import spacy
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor as mlp

file_name = "path-to-pickle-with-labels"
stories = pickle.load(open(file_name, 'rb'))

print(stories[0])

# Required Models for glove
# in case of errors with conda, use this:
# conda install -c conda-forge spacy
# this is what worked for me :P

# !python -m spacy download en
# !python -m spacy download en_core_web_lg
# !python -m spacy link en_core_web_lg en --force

# use the large model as the default model for English textual data

# Initializing the processor
embedder = spacy.load('en')

# basic embeddings using averaged glove vectors
# using Spacy's large language model

def get_embedding(text):
    extract = embedder(text)
    total_sum = np.zeros(300)
    count = 0
    for token in extract:
        count += 1
        total_sum += np.asarray(token.vector)
    return total_sum / count

# creating the inputs and expected outputs
X_train = []
y_train = []
count = 0
for data in stories:
    count += 1
    doc_emb = get_embedding(data['story_text'])
    # use the function of choice to generate the document embedding

    index = 0
    for sentence in data['story']:
        sent_emb = get_embedding(sentence)
        # use the function of choice to generate the sentence embedding

        x = np.concatenate((sent_emb, doc_emb))
        y = data['scores'][index] 
        index += 1

        X_train.append(x)
        y_train.append(y)

    if count > 100:
        break

X_train = np.asmatrix(X_train)
y_train = np.asarray(y_train)

def train(X, y):
    model = mlp(hidden_layer_sizes = (1024, 2048, 1024, 512, 256), max_iter = 100)
    model.fit(X, y)
    return model

def get_values(X, model):
    return model.predict(X)

m = train(X_train, 1000 * y_train)

# Hyperparameter for similarity threshold
theta = 0.95

def similarity(A, B):
    similarity =  (A @ B.T) / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity

def get_top_5(X_doc, y):
    order = np.flip(np.argsort(y))
    sentence_set = []
    for sent_id in order:
        if sentence_set == []:
            sentence_set.append(order[0])
            continue

        consider = X_doc[sent_id, :]
        flag = 1
        for consider_id in sentence_set:
            if similarity(X_doc[consider_id, :], consider) > theta:
                flag = 0
                break

        if flag == 1:
            sentence_set.append(sent_id)
    return sentence_set[0: min(5, len(sentence_set))]

# evaluation
# testing out each document iteratively
# test set: document 950 onwards

doc_id = 950
doc_count = len(stories)

# set the number of documents for testing
limit = 960

while doc_id < min(doc_count, limit):
    X_doc = []
    y_doc = []
    data = stories[doc_id]
    doc_emb = get_embedding(data['story_text'])

    index = 0
    for sentence in data['story']:
        sent_emb = get_embedding(sentence)

        x = np.concatenate((sent_emb, doc_emb))
        y = data['scores'][index] 

        index += 1

        X_doc.append(x)
        y_doc.append(y)

    X_doc = np.asmatrix(X_doc)
    y_doc = np.asarray(y_doc)

    sentence_predicted_scores = get_values(X_doc, m)

    loss = np.linalg.norm(sentence_predicted_scores - y_doc)

    # Uncomment to view the test_loss on the sample  
    # print(loss)

    print("Document ID:", doc_id, ", Top 5 Sentences:", get_top_5(X_doc, sentence_predicted_scores))

    # Uncomment to view the top 10 sentences based on Gold Labels
    # print(np.flip(np.argsort(y_doc))[0:10])
    doc_id += 1

# ^_^ Thank You