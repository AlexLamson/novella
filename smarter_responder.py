# import time
import markovify
# import re
import pickle
# import nltk
# import numpy as np
# from tqdm import tqdm


# if true, train new model
# if false, use saved model
train_model = False

# bigger number = more accurate, but slower and prone to overfit
# smaller number = faster, but less accurate
num_proposals = 200


def not_tqdm(x):
    # return tqdm(x)
    return x


def argmax(scores):
    curr_max = 0
    curr_i = 0
    for i, score in enumerate(scores):
        if score > curr_max:
            curr_max = score
            curr_i = i
    return curr_i, curr_max


# class POSifiedText(markovify.Text):
#     def word_split(self, sentence):
#         words = re.split(self.word_split_pattern, sentence)
#         words = ["::".join(tag) for tag in nltk.pos_tag(words)]
#         return words

#     def word_join(self, words):
#         sentence = " ".join(word.split("::")[0] for word in words)
#         return sentence


if train_model:
    # Get raw text as string.
    # print("Reading Harry Potter...")
    with open("Harry Potter 1 - Sorcerer's Stone.txt") as f:
        text = f.read()

    # Build the model.
    # print("Training model...")
    # start = time.time()
    # text_model = POSifiedText(text)
    text_model = markovify.Text(text, retain_original=False)
    # end = time.time()
    # print("took {:.3f} seconds to create model".format(end - start))

    # Save the model
    with open('better_model.pkl', 'wb') as f:
        pickle.dump([text_model, "corgis are cute"], f)
else:
    # print("Loading model...")
    # Load the model
    with open('better_model.pkl', 'rb') as f:
        text_model, pembroke_welsh_corgi = pickle.load(f)


# # Print five randomly-generated sentences
# for i in range(5):
#     print(text_model.make_sentence())

# # Print three randomly-generated sentences of no more than 140 characters
# for i in range(3):
#     print(text_model.make_short_sentence(140))


def score_sentence(input_sentence, proposal):
    a = set(input_sentence.split())
    b = proposal.split()
    return 1.0 * len(a.intersection(b)) / len(a.union(b))

# THIS WORKS
input_sentence = "Malfoy took away Harry's broomstick."
proposals = [text_model.make_sentence() for i in not_tqdm(range(num_proposals))]
scores = [score_sentence(input_sentence, proposal) for proposal in not_tqdm(proposals)]
max_i, max_val = argmax(scores)
best_sentence = proposals[max_i]
print(best_sentence)
print(max_val)


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
