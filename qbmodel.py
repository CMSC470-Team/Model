from typing import List, Tuple
import nltk
import sklearn
import transformers
import numpy as np
import pandas as pd
import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert import bert_sentence

class QuizBowlModel:

    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.
        Do NOT load your model or resources in the guess_and_buzz() function, 
        as it will increase latency severely. 
        """
        self.embedder = SentenceTransformer('./model1')
        
        self.sentence_embeddings = sentence_embeddings # sentence embeddings need to be unpickled
        self.answer = np.asarray(answers) # answers need to be unpickled
        self.runs = runs ## line 185 and 208 of bert.py needs run_length, this breaks rn
        self.guesser = bert_sentence
        self.buzzer = torch.load('./trained_model.th')

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts a list of question strings, and returns a list of tuples containing
        strings representing the guess and corresponding booleans representing 
        whether or not to buzz. 
        So, guess_and_buzz(["This is a question"]) should return [("answer", False)]
        If you are using a deep learning model, try to use batched prediction instead of 
        iterating using a for loop.
        """
        # get raw guesses
        embeddings = self.embedder.encode(question_text)
        guesses = []
        distance = cosine_similarity(embeddings, self.sentence_embeddings)
        seq = np.argsort(-1*distance, axis=1)
        for i, d in zip(seq, distance):
            guesses.append([])
            sorted_answer = np.take_along_axis(self.answer, i, axis=0)
            for j in range(10):
                guess[len(guesses)-1].append((sorted_answer[j], d[i[j]]))
        
        # generate input for buzzer
        scores = [guess[1] for guess in guesses]
        top_num = [0] * 5
        for i in range(5):
            counter = 0
            percent = 0
            for guess in guesses:
                gg, ss = guess[i]
                gg1, ss1 = guesses[0][i]
                if gg == gg1:
                    percent += ss
                counter += ss
            top_num[i] = percent / counter
        
        # generate input for buzzer
        for raw_guess, rr, num in zip(guesses[0], runs[0], top_num):
            gg, ss = raw_guess
            in_text = False
            if gg in rr:
                in_text = True
            end = False
            if "For 10 points" in rr:
                end = True
            guess = {"guess:%s" % gg: 1,
                     "run_length": float(len(rr) / 1000),
                     "score": float(ss),
                     "label": qq.page == gg,
                     "category:%s" % qq.category: 1,
                     "in_text": in_text,
                     "year:%s" % qq.year: 1,
                     "change": last_guess != gg,
                     "percent": float(num),
                     # "end": end,
                     "freq": float(guesser.get_answer_category_freq(gg, qq.category))
                    }
                last_guess = gg
                ners = get_ner(rr)
                for k, v in ners.items():
                    for vi in v:
                        guess["{}:{}".format(k, vi)] = 1
                for ii in guess:
                    # Don't let it use features that would allow cheating
                    if ii not in censor_features and ii not in vocab:
                        vocab.append(ii)

        # need to figure out how to evaluate using guess
        return 'dones'




# Testing functionality
qbmodel = QuizBowlModel()
print(qbmodel.guess_and_buzz(['This is a question']))