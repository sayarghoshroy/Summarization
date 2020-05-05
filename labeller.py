import pickle

# !pip install rouge
from rouge import Rouge

from google.colab import drive
drive.mount('/content/drive')

file_name = "path-to-pickled-file_:cnn_dataset_1000.pkl"
stories = pickle.load(open(file_name, 'rb'))

print(stories[0])

# Getting the Rouge Scores of Sentences w.r.t the summary
rouge = Rouge()

index = 0

for data in stories:
    story_text = ""
    for sentence in data['story']:
        story_text = story_text + sentence + "."
    # using '.' as the sentence delimiter

    sentence_scores = []
    for sentence in data['story']:
        # scores = rouge.get_scores(sentence, summary)
        # sentence_scores.append(scores['rouge-2']['f'])
        # issues arise due to maximum permissible length allowed in ROUGE-L implementation
        
        # getting the total rouge score as an average of sentence-wise ROUGE-scores with
        # all sentences in the document

        if len(sentence) < 2:
                continue
        total = 0
        for comparison in data['highlights']:
            if len(sentence) < 2 or len(comparison) < 2:
                continue
            try:
                scores = rouge.get_scores(sentence, comparison)
            except Exception as e:
                # print(e)
                # print("Error Occured at Index: " + str(index))
                # errors arise due to cases where the comparison took place with '.' or '...'
                # we can safely ignore all of that
                continue
            total += scores[0]['rouge-2']['f']
        sentence_scores.append(total / len(data['story']))
    data['story_text'] = story_text
    data['scores'] = sentence_scores
    index += 1

print(stories[343])

from pickle import dump
dump(stories, open('cnn_dataset_1000_labelled.pkl', 'wb'))

# ^_^ Thank You