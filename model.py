import pickle
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor as mlp
# from IPython.display import display

# import warnings
# warnings.filterwarnings('ignore')

# !pip install rouge
from rouge import Rouge

# from google.colab import drive
# drive.mount('/content/drive')

# change to path to dataset
file_name = "/content/drive/My Drive/Summarization_Pickled_Data/cnn_dataset_1000_labelled.pkl"
stories = pickle.load(open(file_name, 'rb'))

# displaying the first datapoint
# verify correctness of load

# Uncomment to Display the First Datapoint
# print(stories[0])

# Required Models for glove
# in case of errors with conda, use this:
# conda install -c conda-forge spacy
# this is what worked for me :P

# uncomment the next two lines if model data cannot be located
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

    for count in tqdm(range(size)):
        data = stories[start_index + count]

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

def make_parameters(train_size):
    batch_size = int(4 * np.sqrt(train_size))
    n_batches = int(32 * (train_size / batch_size))
    # can set batch_size to standard values such as 64, 128, 256

    print("Total Number of Training Examples: " + str(train_size))
    print("Batch Size: " + str(batch_size))
    print("Number of Batches: " + str(n_batches))

    return batch_size, n_batches

def train(X_train, y_train, batch_size, n_batches):
    model = mlp(hidden_layer_sizes = (1024, 2048, 1024, 512, 256, 256, 128, 64), max_iter = 10000)
    
    train_size = np.shape(X_train)[0]

    min_loss = 1e20

    for iterator in tqdm(range(n_batches)):
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

    final_model = pickle.load(open(model_name + '_best_model', 'rb'))
    return final_model

batch_size, n_batches = make_parameters(train_size)

m = train(X_train, 1000 * y_train, batch_size, n_batches)

# Hyperparameter for similarity threshold
theta = 0.95

def similarity(A, B):
    similarity =  (A @ B.T) / (np.linalg.norm(A) * np.linalg.norm(B))
    return similarity

def get_top_k(X_doc, y, k):
    # k should be in {3, 4, 5}
    # error handling
    k = int(k)
    if k > 5:
        k = 5
    elif k < 3:
        k = 3
    
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
    return sentence_set[0: min(k, len(sentence_set))]

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
# to access the final created summary

# set the number of documents for testing
limit = test_size

result = {}
result['3'] = np.zeros(9)
result['4'] = np.zeros(9)
result['5'] = np.zeros(9)
# averaging the ROUGE Metrics
# for different summary lengths

count = 0
all_summaries = []

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

    gold_summary = join(data['highlights'])

    for k in [3, 4, 5]:
        summary_sent_id = get_top_k(X_doc, sentence_predicted_scores, k)
        # Uncomment to view the indices of chosen sentences
        # print("Document ID:", start_doc_id + count, ", Top 5 Sentences:", summary_sent_id)

        # Uncomment to view the top 10 sentences based on Gold Labels
        # print("Top 10 sentences based on Gold Label", np.ndarray.tolist(np.flip(np.argsort(y_doc))[0:10]))

        generated_summary = join([data['story'][idx] for idx in summary_sent_id])

        scores = rouge.get_scores(generated_summary, gold_summary)[0]
        result[str(k)] += extract_rouge(scores)

    summary_eval = {'doc': data['story_text'], 'gen_summ': generated_summary, 'true_summ': gold_summary}
    all_summaries.append(summary_eval)

    count += 1

for k in [3, 4, 5]:
    result[str(k)] = result[str(k)] / test_size

predicted = get_values(X_test, m)
test_loss = get_loss(y_test, predicted)

print("Sample Output:")
print("Document:\n", stories[-1]['story_text'])
print("Generated Summary:\n", generated_summary)
print("Gold Summary:\n", gold_summary)

print("\nAll Metrics:\n")

data = []
for k in [3, 4, 5]:
    lst = np.ndarray.tolist(result[str(k)])
    lst.append(test_loss)
    data.append(lst)

df = pd.DataFrame(data, columns = ['R1-f', 'R1-p', 'R1-r',
                                    'R2-f', 'R2-p', 'R2-r',
                                    'Rl-f', 'Rl-p', 'Rl-r',
                                    'Loss'], dtype = float)

df.index = ['glove top-3', 'glove top-4', 'glove top-5']
display(df)

# save results into a dataframe file
df.to_csv(model_name + '_results.csv')

# verifying creation of summaries
# uncomment to display the first summary
# print(all_summaries[0])

filename = model_name + 'summaries_eval.pickle'
# dumping summaries into a pickle file for further loading and evaluation
with open(filename, 'wb') as f:
    pickle.dump(all_summaries, f)

# verifying pickled file

pickle_in = open(filename, "rb")
eval_summ = pickle.load(pickle_in)

# uncomment to display the second summary
# print(all_summaries[1])

# ^_^ Thank You