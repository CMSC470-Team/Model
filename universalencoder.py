import tensorflow_hub as hub 
import tensorflow as tf
from typing import List, Optional, Tuple
from collections import defaultdict
import json
import argparse
from typing import Union, Dict
import numpy as np
from qanta_util.qbdata import QantaDatabase
from tfidf_guesser_test import StubDatabase
import nltk
from sgd import kBIAS
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')
MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3
MODULE_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'


def get_ner(text: str):
    d = defaultdict(lambda: [])
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                label = chunk.label()
                in_txt_names = " ".join([c[0] for c in chunk])
                d[label].append(in_txt_names)
    return d


class bert_sequence:

    def __init__(self):
        self.sbert_model = hub.load(MODULE_URL)
        self.sentence_embeddings = None
        self.answer = None
        self.category_freq = dict()
        self.category_answer_freq = dict()

    def get_answer_category_freq(self, answer, category):
        freq = 0
        if answer in self.category_answer_freq.keys():
            if category in self.category_answer_freq[answer].keys():
                freq = self.category_answer_freq[answer][category]/self.category_freq[answer]
        return freq

    def train(self, training_data: Union[StubDatabase, QantaDatabase], limit=-1) -> None:
        category = [x.category for x in training_data.guess_train_questions]
        questions = [x.text for x in training_data.guess_train_questions]
        answers = [x.page for x in training_data.guess_train_questions]
        sentences = []
        for text in questions:
            text = text.replace('\n', ' ')
            sentences.append(re.sub("[.,!?\\-]", '', text.lower()))
        sentence_embeddings = self.sbert_model(sentences)
        self.sentence_embeddings = sentence_embeddings
        self.answer = np.asarray(answers)

        p_dict = dict()
        for c,a in zip(category,answers):
            if a in p_dict.keys():
                if c in p_dict[a].keys():
                    p_dict[a][c] += 1
                else:
                    p_dict[a][c] = 0
            else:
                p_dict[a] = dict()
        f_dict = dict()
        for c in answers:
            if c in f_dict.keys():
                f_dict[c] += 1
            else:
                f_dict[c] = 0

        self.category_freq = f_dict
        self.category_answer_freq = p_dict

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        query = []
        for text in questions:
            text = text.replace('\n', ' ')
            query.append(re.sub("[.,!?\\-]", '', text.lower()))
        guesses = []
        query_vec = self.sbert_model(query)
        distance = cosine_similarity(query_vec, self.sentence_embeddings)
        seq = np.argsort(-1*distance, axis=1)
        for i, d in zip(seq, distance):
            guesses.append([])
            sorted_answer = np.take_along_axis(self.answer, i, axis=0)
            for j in range(max_n_guesses):
                guesses[len(guesses)-1].append((sorted_answer[j], d[i[j]]))
        return guesses

    def confusion_matrix(self, evaluation_data: QantaDatabase, limit=-1) -> Dict[str, Dict[str, int]]:
        questions = [x.text for x in evaluation_data.guess_dev_questions]
        answers = [x.page for x in evaluation_data.guess_dev_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        d = defaultdict(dict)
        data_index = 0 #sol
        guesses = [x[0][0] for x in self.guess(questions, max_n_guesses=1)] #sol
        for gg, yy in zip(guesses, answers): #sol
            d[yy][gg] = d[yy].get(gg, 0) + 1 #sol
            data_index += 1 #sol
            if data_index % 100 == 0: #sol
                print("%i/%i for confusion matrix" % (data_index, #sol
                                                      len(guesses))) #sol
        return d


def write_guess_json(guesser, filename, fold, run_length=200, censor_features=["id", "label"]):
    vocab = [kBIAS]

    print("Writing guesses to %s" % filename)
    num = 0
    with open(filename, 'w') as outfile:
        total = len(fold)
        for qq in fold:
            num += 1
            if num % (total // 80) == 0:
                print('.', end='', flush=True)
            last_guess = ""
            runs = qq.runs(run_length)
            guesses = guesser.guess(runs[0], max_n_guesses=5)
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

                outfile.write(json.dumps(guess, sort_keys=True))
                outfile.write("\n")
    print("")
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--guesstrain", default="data/small.guesstrain.json", type=str)
    parser.add_argument("--guessdev", default="data/small.guessdev.json", type=str)
    parser.add_argument("--buzztrain", default="data/small.buzztrain.json", type=str)
    parser.add_argument("--buzzdev", default="data/small.buzzdev.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--vocab", default="", type=str)
    parser.add_argument("--buzztrain_predictions", default="", type=str)
    parser.add_argument("--buzzdev_predictions", default="", type=str)

    flags = parser.parse_args()

    print("Loading %s" % flags.guesstrain)
    guesstrain = QantaDatabase(flags.guesstrain)
    guessdev = QantaDatabase(flags.guessdev)

    bert_guesser = bert_sequence()
    bert_guesser.train(guesstrain, limit=flags.limit)

    confusion = bert_guesser.confusion_matrix(guessdev, limit=-1)
    print("Errors:\n=================================================")
    for ii in confusion:
        for jj in confusion[ii]:
            if ii != jj:
                print("%i\t%s\t%s\t" % (confusion[ii][jj], ii, jj))


    if flags.buzztrain_predictions:
        print("Loading %s" % flags.buzztrain)
        buzztrain = QantaDatabase(flags.buzztrain)
        vocab = write_guess_json(bert_guesser, flags.buzztrain_predictions, buzztrain.buzz_train_questions)

    if flags.vocab:
        with open(flags.vocab, 'w', encoding = 'utf8') as outfile:
            for ii in vocab:
                outfile.write("%s\n" % ii)

    if flags.buzzdev_predictions:
        assert flags.buzztrain_predictions, "Don't have vocab if you don't do buzztrain"
        print("Loading %s" % flags.buzzdev)
        buzzdev = QantaDatabase(flags.buzzdev)
        write_guess_json(bert_guesser, flags.buzzdev_predictions, buzzdev.buzz_dev_questions)