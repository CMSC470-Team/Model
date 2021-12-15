from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from numpy import zeros, sign
import lr_pytorch

def input_to_buzzer(question_guesses, question_text, vocab):
    input = []
    for gueses, question in zip(question_guesses, question_text):
        run_length = float(len(question) / 1000)
        guess = "guess:"+gueses[0][0]
        g = gueses[0][0]
        score = gueses[0][1]
        in_text = False
        if g in question:
            in_text = True
        sums = 0
        total = 0
        for guess1 in gueses:
            if guess1[0] == g:
                sums += guess1[1]
            total += guess1[1]
        percent = float(sums/total)
        x = zeros(len(vocab))
        x[vocab.index("run_length")] += run_length
        x[vocab.index(guess)] = 1
        x[vocab.index("score")] += score
        x[vocab.index("percent")] += percent
        if in_text:
            x[vocab.index("in_text")] == 1
        input.append(x)
    input = np.array(input)
    input = torch.from_numpy(input)
    input = input.type(torch.FloatTensor)
    return input


class QuizBowlModel:

    def __init__(self):
        """
		Load your model(s) and whatever else you need in this function.

		Do NOT load your model or resources in the guess_and_buzz function,
as it will increase latency severely.
		"""
        #self.sbert_model = SentenceTransformer('./model1')
        self.sbert_model = SentenceTransformer('./model5')
        with open("bert2.vocab", 'r', encoding='utf8') as infile:
            self.vocab = [x.strip() for x in infile]
        model = lr_pytorch.SimpleLogreg(len(self.vocab))
        model.load_state_dict(torch.load("trained_model.th"))
        model.eval()
        self.buzzer = model
        with open('link_score3.pickle', 'rb') as handle:
            self.sentence_embeddings = pickle.load(handle)
        with open('answers3.pickle', 'rb') as handle:
            self.answer = pickle.load(handle)
        self.answer = np.asarray(self.answer)

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
		This function accepts list of question string, and returns a string
		representing the guess and a corresponding boolean representing
		whether or not to buzz.

		If you are using a deep learning model, try to
		use batched prediction instead of iterating using a for loop.
		"""
        question_guesses = self.guess(question_text, 5)
        buzz_input = input_to_buzzer(question_guesses, question_text, self.vocab)
        buzz_output = self.buzzer(buzz_input)
        buzz_output = torch.round(buzz_output.data).tolist()
        outputs = []
        for guesses, buzz in zip(question_guesses,buzz_output):
            outputs.append((guesses[0][0], int(buzz[0])==1))
        return outputs

    def guess(self, questions, max_n_guesses):
        query = []
        for text in questions:
            text = text.replace('\n', ' ')
            query.append(re.sub("[.,!?\\-]", '', text.lower()))
        guesses = []
        query_vec = self.sbert_model.encode(query)
        distance = cosine_similarity(query_vec, self.sentence_embeddings)
        seq = np.argsort(-1*distance, axis=1)
        for i, d in zip(seq, distance):
            guesses.append([])
            sorted_answer = np.take_along_axis(self.answer, i, axis=0)
            for j in range(max_n_guesses):
                guesses[len(guesses)-1].append((sorted_answer[j], d[i[j]]))
        return guesses

QuizBowlModel = QuizBowlModel()
print(QuizBowlModel.guess_and_buzz(["The Pindus Mountains in this country are home to six monasteries built on sandstone rock pillars known collectively as the Meteora. \"The Gates\" is a four-meter wide passage in this country's Samaria Gorge, which lies on the same island as the city of Heraklion. Its Thracian city of (*) Thessaloniki is the capital of a region that shares its name with this country's northern neighbor. Corfu is among its islands in the Ionian Sea. The Propylaea, the Erechtheion, and the Parthenon are among the ancient structures atop the Acropolis in this country's capital. For ten points, identify this country at the tip of the Balkan Peninsula, with capital at Athens.", "The Pindus Mountains in this country are home to six monasteries built on sandstone rock pillars known collectively as the Meteora."]))
