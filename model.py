import pickle
import spacy
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor as mlp
from IPython.display import display

# import warnings
# warnings.filterwarnings('ignore')

# !pip install rouge
from rouge import Rouge

from google.colab import drive
drive.mount('/content/drive')

# change to path to dataset
file_name = "/content/drive/My Drive/Summarization_Pickled_Data/cnn_dataset_1000_labelled.pkl"
stories = pickle.load(open(file_name, 'rb'))

# displaying the first datapoint
# verify correctness of load
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
train_size = 900
val_size = 50
test_size = 50

def make_set(start_index, size):
    count = 0
    X_set = []
    y_set = []

    while count < size:
        data = stories[start_index + count]
        count += 1

        doc_emb = get_embedding(data['story_text'])
        # use the function of choice to generate the document embedding

        index = 0
        for sentence in data['story']:
            sent_emb = get_embedding(sentence)
            # use the function of choice to generate the sentence embedding

            x = np.concatenate((sent_emb, doc_emb))
            try:
                y = data['scores'][index]
            except:
                y = 0.0
            index += 1

            X_set.append(x)
            y_set.append(y)

    return np.asmatrix(X_set), np.asarray(y_set)

X_train, y_train = make_set(0, train_size)
X_val, y_val = make_set(train_size, val_size)
X_test, y_test = make_set(train_size + val_size, test_size)

def get_values(X, model):
    return model.predict(X)

def get_loss(pred, y):
    return np.linalg.norm(pred - y) / np.shape(y)[0]

model_name = "glove_averaged"
# modify the model name

def train(X_train, y_train):
    model = mlp(hidden_layer_sizes = (1024, 2048, 1024, 512, 512, 256, 128), max_iter = 1000)
    
    train_size = np.shape(X_train)[0]

    batch_size = int(np.sqrt(train_size))
    n_batches = int(4 * (train_size / batch_size))

    print("Total Number of Training Examples: " + str(train_size))
    print("Batch Size: " + str(batch_size))
    print("Number of Batches: " + str(n_batches))

    min_loss = 1e20

    while(n_batches > 0):
        idx = np.random.randint(0, train_size, size = batch_size)

        X_select = X_train[idx,:]
        y_select = y_train[idx]

        model.partial_fit(X_select, y_select)

        sentence_predicted_scores = get_values(X_val, model)

        loss = get_loss(sentence_predicted_scores, y_val)

        # saving best model seen so far
        if loss < min_loss:
            min_loss = loss
            pickle.dump(model, open(model_name + '_best_model', 'wb'))

        n_batches -= 1

    final_model = pickle.load(open(model_name + '_best_model', 'rb'))
    return final_model

m = train(X_train, 1000 * y_train)

# Hyperparameter for similarity threshold
theta = 0.95

def similarity(A, B):
    similarity =  (A @ B.T) / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity

def get_top_4(X_doc, y):
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
    return sentence_set[0: min(4, len(sentence_set))]

# Creating object of the ROUGE class
rouge = Rouge()

# evaluation
# testing out each document iteratively
# test set: document 'train_size + val_size' onwards

def join(lst):
    string = ""
    for elem in lst:
        string = string + elem + " . "
    return string

def extract_rouge(rouge_dict):
    scores = []

    scores.append(100 * rouge_dict["rouge-1"]['f'])
    scores.append(100 * rouge_dict["rouge-1"]['p'])
    scores.append(100 * rouge_dict["rouge-1"]['r'])

    scores.append(100 * rouge_dict["rouge-2"]['f'])
    scores.append(100 * rouge_dict["rouge-2"]['p'])
    scores.append(100 * rouge_dict["rouge-2"]['r'])

    scores.append(100 * rouge_dict["rouge-l"]['f'])
    scores.append(100 * rouge_dict["rouge-l"]['p'])
    scores.append(100 * rouge_dict["rouge-l"]['r'])

    return np.asarray(scores)

start_doc_id = train_size + val_size
doc_count = len(stories)

generated_summary, gold_summary = 0, 0

# set the number of documents for testing
limit = test_size

total = np.zeros(9)
# averaging the 9 ROUGE Metrics

count = 0

while count < min(doc_count, limit):
    X_doc = []
    y_doc = []
    data = stories[start_doc_id + count]
    doc_emb = get_embedding(data['story_text'])

    index = 0
    for sentence in data['story']:
        sent_emb = get_embedding(sentence)

        x = np.concatenate((sent_emb, doc_emb))
        try:
            y = data['scores'][index]
        except:
            y = 0.0

        index += 1

        X_doc.append(x)
        y_doc.append(y)

    X_doc = np.asmatrix(X_doc)
    y_doc = np.asarray(y_doc)

    sentence_predicted_scores = get_values(X_doc, m)

    loss = np.linalg.norm(sentence_predicted_scores - y_doc)

    # Uncomment to view the test_loss on the sample  
    # print(loss)

    summary_sent_id = get_top_4(X_doc, sentence_predicted_scores)
    # Uncomment to view the indices of chosen sentences
    # print("Document ID:", start_doc_id + count, ", Top 5 Sentences:", summary_sent_id)

    # Uncomment to view the top 10 sentences based on Gold Labels
    # print("Top 10 sentences based on Gold Label", np.ndarray.tolist(np.flip(np.argsort(y_doc))[0:10]))

    gold_summary = join(data['highlights'])
    generated_summary = join([data['story'][idx] for idx in summary_sent_id])

    scores = rouge.get_scores(generated_summary, gold_summary)[0]
    total += extract_rouge(scores)

    count += 1

averaged = total / test_size

predicted = get_values(X_test, m)
test_loss = get_loss(y_test, predicted)

print("Sample Output:")
print("Document:\n", stories[-1]['story_text'])
print("Generated Summary:\n", generated_summary)
print("Gold Summary:\n", gold_summary)

print("All Metrics:")

lst = np.ndarray.tolist(averaged)
lst.append(test_loss)

print()

df = pd.DataFrame([lst], columns = ['R1-f', 'R1-p', 'R1-r',
                                    'R2-f', 'R2-p', 'R2-r',
                                    'Rl-f', 'Rl-p', 'Rl-r',
                                    'Test Regression Loss'], dtype = float)
df.index = ['Averaged Glove Vectors']
display(df)

# save results into a dataframe file
df.to_csv(model_name + '_results.csv')

# ^_^ Thank You