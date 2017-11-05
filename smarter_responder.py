# import time
import markovify
# import re
import pickle
# import nltk
# import numpy as np

'''
Notable samples
===============
Harry's legs were like baby dolphins.
Harry felt the grin slide off his broomstick.
His eyes were fixed upon Harry's eardrums.
Voldemort himself created his worst subject.
'''

# if true, train new model
# if false, use saved model
train_model = False

# bigger number = more accurate, but slower and prone to overfit
# smaller number = faster, but less accurate
num_proposals = 135

use_multiple_books = True


if use_multiple_books:
    filenames = """Harry Potter 1 - Sorcerer's Stone.txt
Harry Potter 2 - Chamber of Secrets.txt
Harry Potter 3 - The Prisoner Of Azkaban.txt
Harry Potter 4 - The Goblet Of Fire.txt
Harry Potter 5 - Order of the Phoenix.txt
Harry Potter 6 - The Half Blood Prince.txt
Harry Potter 7 - Deathly Hollows.txt""".split('\n')
else:
    filenames = """Harry Potter 1 - Sorcerer's Stone.txt""".split('\n')


stopwords = """a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours  ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split()

def argmax(scores):
    curr_max = 0
    curr_i = 0
    for i, score in enumerate(scores):
        if score > curr_max:
            curr_max = score
            curr_i = i
    return curr_i


# class POSifiedText(markovify.Text):
#     def word_split(self, sentence):
#         words = re.split(self.word_split_pattern, sentence)
#         words = ["::".join(tag) for tag in nltk.pos_tag(words)]
#         return words

#     def word_join(self, words):
#         sentence = " ".join(word.split("::")[0] for word in words)
#         return sentence


if train_model:
    from tqdm import tqdm
    print("training models")

    my_lovely_models = []

    for filenum, filename in tqdm(enumerate(filenames), total=len(filenames)):
        # load the book into memory
        with open(filename) as f:
            text = f.read()

        # train markov chain model on it
        text_model = markovify.Text(text, retain_original=False, state_size = 3)

        # add model to list of models
        my_lovely_models += [text_model]

        # with open('model_{}.pkl'.format(filenum), 'wb') as f:
        #     pickle.dump([text_model, "corgis are cute"], f)

    print("merging models")
    uber_model = markovify.combine(my_lovely_models)
    with open('multi_book_model.pkl', 'wb') as f:
        pickle.dump([uber_model, "corgis are cute"], f)

else:
    # Load the model
    if use_multiple_books:
        with open('multi_book_model.pkl', 'rb') as f:
            text_model, pembroke_welsh_corgi = pickle.load(f)
    else:
        with open('better_model.pkl', 'rb') as f:
            text_model, pembroke_welsh_corgi = pickle.load(f)


# # Print five randomly-generated sentences
# for i in range(5):
#     print(text_model.make_sentence())

# # Print three randomly-generated sentences of no more than 140 characters
# for i in range(3):
#     print(text_model.make_short_sentence(140))

# don't use stop words in the distance function
def filter_sentence(sentence):
    return " ".join([word for word in sentence.split() if (word.lower() not in stopwords)])

# scores a proposal sentence against a previous set of sentences for relevancy
def score_sentence(input_sentences, proposal):
    mask = [1.0, 1.0, 0.75, 0.75, 0.6525, 0.6525]

    total = 0
    for i, (sentence, creator) in enumerate(input_sentences):
        total += _score_sentence(sentence, proposal) * mask[i] * (1 if creator is 'human' else 1.0/3)

# helper for score sentence, scores a proposal against a single input sentence
def _score_sentence(input_sentence, proposal):
    input_sentence = input_sentence.replace(r'[^\w ]+', ' ').replace(r'\s+', ' ').strip()
    proposal = proposal.replace(r'[^\w ]+', ' ').replace(r'\s+', ' ').strip()

    a = set(filter_sentence(input_sentence).split())
    b = set(filter_sentence(proposal).split())
    score = 1.0 * len(a.intersection(b)) / len(a.union(b))
    if len(b) < 4:
        score -= (1.0 / len(b))
    if len(b) > 12:
        score -= 0.05*len(b)
        # score -= 0.2
    return score
    # denom = max(len(a), len(b))
    # # return 1.0 * len(a.intersection(b)) / denom
    # return 1.0 * len(a.intersection(b))

previous_sentences = []
max_previous_sentences = 6

# THIS WORKS
input_sentence = "Malfoy took away Harry's broomstick."
previous_sentences += [(input_sentence, 'human')]
print(input_sentence)
for _ in range(10):
    proposals = [text_model.make_sentence() for i in range(num_proposals)]
    scores = [score_sentence(previous_sentences, proposal) for proposal in proposals]
    best_sentence = proposals[argmax(scores)]

    previous_sentences.insert(0, (best_sentence, 'bot'))
    if (len(previous_sentences) > max_previous_sentences):
        previous_sentences.pop() # previous_sentences = previous_sentences[:-6]
    print(best_sentence)


# def make_sentence(input_sentence):
#     proposals = [text_model.make_sentence() for i in range(num_proposals)]

#     scores = [score_sentence(input_sentence, proposal) for proposal in proposals]
#     best_sentence = proposals[np.argmax(scores)]
#     return best_sentence


# input_sentence = "Malfoy took away Harry's broomstick."
# curr_sentence = input_sentence
# print(curr_sentence)
# for _ in range(10):
#     curr_sentence = make_sentence(curr_sentence)
#     print(curr_sentence)
