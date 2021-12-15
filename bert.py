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
from sentence_transformers import SentenceTransformer, InputExample, evaluation, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import pickle

nltk.download('wordnet')
MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 100
BUZZ_THRESHOLD = 0.3


def find_k_top_related_entities(entities_list1, entities_list2, score_dict):
    non_found1 = []
    non_found2 = []
    found1 = []
    found2 = []
    pair_list1 = []
    for e in entities_list1:
        if e in score_dict:
            found1.append(e)
            scores = score_dict[e]
            for i in scores:
                pair_list1.append(i)
        else:
            non_found1.append(e)
    pair_list2 = []
    for e in entities_list2:
        if e in score_dict:
            found2.append(e)
            scores = score_dict[e]
            for i in scores:
                pair_list2.append(i)
        else:
            non_found2.append(e)

    k = len(found1)+len(found2)
    z = len(pair_list1) + len(pair_list2)
    s = len(non_found1)+len(non_found2)
    if (k+s)==0:
        return -1
    ratio = k/(k+s)
    proportion = 0
    if s != 0:
        same_list = list(set(non_found1).intersection(non_found2))
        proportion += len(same_list)*2/(s)
    if k == 0:
        return proportion*(1-ratio)

    search_dict = dict()
    count_dict = dict()
    search_dict1 = dict()
    count_dict1 = dict()
    count = 0
    for i in pair_list2:
        if i[0] in search_dict:
            search_dict[i[0]] = search_dict[i[0]]+i[1]
        else:
            search_dict[i[0]] = i[1]
    for i in pair_list2:
        if i[0] in count_dict:
            count_dict[i[0]] += 1
        else:
            count_dict[i[0]] = 1
    for key in search_dict:
        search_dict[key] = search_dict[key]/count_dict[key]
    for i in pair_list1:
        if i[0] in search_dict1:
            search_dict1[i[0]] = search_dict1[i[0]]+i[1]
        else:
            search_dict1[i[0]] = i[1]
    for i in pair_list1:
        if i[0] in count_dict1:
            count_dict1[i[0]] += 1
        else:
            count_dict1[i[0]] = 1
    for key in search_dict1:
        search_dict1[key] = search_dict1[key]/count_dict1[key]
    for key, value in search_dict1.items():
        if key in search_dict:
            count += 2*(1-(abs(value-search_dict[key]))**2)
    return proportion*(1-ratio)+(count/z)*ratio

def get_ner(text: str):
    d = defaultdict(lambda: [])
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                label = chunk.label()
                in_txt_names = " ".join([c[0] for c in chunk])
                d[label].append(in_txt_names)
    return d


class bert_sentence:

    def __init__(self):
        self.sbert_model = SentenceTransformer('./model5')
        self.sentence_embeddings = None
        self.answer = None
        self.category_freq = dict()
        self.category_answer_freq = dict()
        self.training_size = 10000

    def get_label(self, p_list, category_list, page_list, q_list, questions):
        with open("entity_link_score.pickle", "rb") as fp:
            score_dict = pickle.load(fp)
        j = 0
        np.random.shuffle(p_list)
        pair_list = np.random.choice(p_list, size=(self.training_size, 2), replace=True)
        train_examples = []
        z = 0
        for i in pair_list:
            z += 1
            if (z%150 == 0):
                print("In progress: ", z/(self.training_size*0.01))
            label = 0
            entities_list1 = []
            entities_list2 = []
            ners = get_ner(questions[i[0]])
            for k, v in ners.items():
                for vi in v:
                    entities_list1.append(vi)
            ners = get_ner(questions[i[1]])
            for k, v in ners.items():
                for vi in v:
                    entities_list2.append(vi)
            entity_score = find_k_top_related_entities(entities_list1, entities_list2, score_dict)
            #if category_list[i[0]] == category_list[i[1]]:
            #    label += 0.2
            label += 0.4*entity_score
            if page_list[i[0]] == page_list[i[1]]:
                label += 0.6
            train_examples.append(InputExample(texts=[q_list[i[0]],q_list[i[1]]], label=label))
        np.random.shuffle(p_list)
        pair_list = np.random.choice(p_list, size=(self.training_size, 2), replace=True)
        label_list = []
        question_list1 = []
        question_list2 = []
        z = 0
        for i in pair_list:
            z += 1
            if (z%150 == 0):
                print("In progress: ", z/(self.training_size*0.01))
            label = 0
            entities_list1 = []
            entities_list2 = []
            ners = get_ner(questions[i[0]])
            for k, v in ners.items():
                for vi in v:
                    entities_list1.append(vi)
            ners = get_ner(questions[i[1]])
            for k, v in ners.items():
                for vi in v:
                    entities_list2.append(vi)
            entity_score = find_k_top_related_entities(entities_list1, entities_list2, score_dict)
            #if category_list[i[0]] == category_list[i[1]]:
            #    label += 0.2
            label += 0.4*entity_score
            if page_list[i[0]] == page_list[i[1]]:
                label += 0.6
            label_list.append(label)
            question_list1.append(q_list[i[0]])
            question_list2.append(q_list[i[1]])
        evaluator = evaluation.EmbeddingSimilarityEvaluator(question_list1, question_list2, label_list)
        while j < int(self.training_size/100):
            np.random.shuffle(p_list)
            pair_list = np.random.choice(p_list, size=(self.training_size, 2), replace=True)
            for i in pair_list:
                if page_list[i[0]] == page_list[i[1]]:
                    print(page_list[i[0]])
                    label = 0
                    entities_list1 = []
                    entities_list2 = []
                    ners = get_ner(questions[i[0]])
                    for k, v in ners.items():
                        for vi in v:
                            entities_list1.append(vi)
                    ners = get_ner(questions[i[1]])
                    for k, v in ners.items():
                        for vi in v:
                            entities_list2.append(vi)
                    entity_score = find_k_top_related_entities(entities_list1, entities_list2, score_dict)
                #    if category_list[i[0]] == category_list[i[1]]:
                #            label += 0.2
                    label += 0.4*entity_score
                    j += 1
                    label += 0.6
                    train_examples.append(InputExample(texts=[q_list[i[0]],q_list[i[1]]], label=label))
                    #label_list.append(label)
                    #question_list1.append(q_list[i[0]])
                    #question_list2.append(q_list[i[1]])
        score_dict.clear()
        return train_examples, evaluator


    def get_answer_category_freq(self, answer, category):
        freq = 0
        if answer in self.category_answer_freq.keys():
            if category in self.category_answer_freq[answer].keys():
                freq = self.category_answer_freq[answer][category]/self.category_freq[answer]
        return freq

    def train(self, training_data: Union[StubDatabase, QantaDatabase], limit=-1) -> None:
        category = [x.category for x in training_data.guess_train_questions]
        pair_list = np.asarray(list(range(len(category))))
        questions = [x.text for x in training_data.guess_train_questions]
        answers = [x.page for x in training_data.guess_train_questions]
        sentences = []
        for text in questions:
            text = text.replace('\n', ' ')
            sentences.append(re.sub("[.,!?\\-]", '', text.lower()))
        #train_examples, evaluator = self.get_label(pair_list, category, answers, sentences, questions)
        #train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
        #train_loss = losses.CosineSimilarityLoss(model=self.sbert_model)
        #self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
        #  evaluator=evaluator,
        #  epochs=10,
        #  evaluation_steps=500,
        #  warmup_steps=100,
        #  output_path='./model5/')
        sentence_embeddings = self.sbert_model.encode(sentences)
        self.sentence_embeddings = sentence_embeddings
        with open('link_score3.pickle', 'wb') as handle:
            pickle.dump(self.sentence_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #with open('questions.pickle', 'wb') as handle:
        #    pickle.dump(self.questions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('answers3.pickle', 'wb') as handle:
            pickle.dump(answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
        query_vec = self.sbert_model.encode(query)
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
    nums = 0
    with open(filename, 'w') as outfile:
        total = len(fold)
        for qq in fold:
            nums += 1
            if nums % (total // 80) == 0:
                print('.', end='', flush=True)
            if nums % 100 == 0:
                print(nums)
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
                         #"category:%s" % qq.category: 1,
                         "in_text": in_text,
                         #"year:%s" % qq.year: 1,
                         #"change": last_guess != gg,
                         "percent": float(num),
                         # "end": end,
                         #"freq": float(guesser.get_answer_category_freq(gg, qq.category))
                         }
               # last_guess = gg
                #ners = get_ner(rr)
               # for k, v in ners.items():
                #    for vi in v:
                 #       guess["{}:{}".format(k, vi)] = 1
                for ii in guess:
                    # Don't let it use features that would allow cheating
                    if ii not in censor_features and ii not in vocab:
                        vocab.append(ii)

                outfile.write(json.dumps(guess, sort_keys=True))
                outfile.write("\n")
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--guesstrain", default="data/qanta.train.2018.04.18.json", type=str)
    parser.add_argument("--guessdev", default="data/qanta.dev.2018.04.18.json", type=str)
    parser.add_argument("--buzztrain", default="data/qanta.train.2018.04.18.json", type=str)
    parser.add_argument("--buzzdev", default="data/qanta.dev.2018.04.18.json", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--vocab", default="", type=str)
    parser.add_argument("--buzztrain_predictions", default="", type=str)
    parser.add_argument("--buzzdev_predictions", default="", type=str)
    parser.add_argument("--model", default="BERT", type=str)

    flags = parser.parse_args()

    print("Loading %s" % flags.guesstrain)
    guesstrain = QantaDatabase(flags.guesstrain)
    guessdev = QantaDatabase(flags.guessdev)

    bert_guesser = bert_sentence()

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
