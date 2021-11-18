# Seyed Mohammad Ghaemi

import random
from numpy import zeros, sign
import numpy as np
from math import exp, log
from collections import defaultdict
import json
#import matplotlib.pyplot as plt

import argparse

from numpy.core.fromnumeric import transpose

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Note: Prevents overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    # You'll want to use this function, but don't modify it
    
    if abs(score) > threshold:
        score = threshold * sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, json_line, vocab, use_bias=True):
        """
        Create a new example

        label -- The label (0 / 1) of the example
        words -- The words in a list of "word:count" format
        vocab -- The vocabulary to use as features (list)
        use_bias -- Include a bias feature (should be false with Pytorch)
        """

        # Use but don't modify this function
        
        self.nonzero = {}
        self.y = 1 if json_line["label"] else 0
        self.x = zeros(len(vocab))

        for feature in json_line:
            if feature in vocab:
                assert feature != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(feature)] += float(json_line[feature])
                self.nonzero[vocab.index(feature)] = feature
        # Initialize the bias feature
        if use_bias:
            self.x[0] = 1


class LogReg:
    def __init__(self, num_features, learning_rate=0.001):
        """
        Create a logistic regression classifier

        num_features -- The number of features (including bias)
        learning_rate -- How big of a SG step we take
        """

        # You *may* want to add additional data members here
        
        self.beta = zeros(num_features)
        self.learning_rate = learning_rate

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy, which is returned as a tuple.

        examples -- The dataset to score
        """

        # You probably don't need to modify this code
        # TAcc is training acc
        # HAcc is heldout (test) acc
        # TProb and zHProb is the log of the product of all the prob. 
        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example):
        """
        Compute a stochastic gradient update to improve the log likelihood and return the new feature weights.

        train_example -- The example to take the gradient with respect to
        """

        dot_val = self.beta @ train_example.x.T
        pr = np.exp(dot_val) / (1 + np.exp(dot_val))
        
        self.beta = self.beta + self.learning_rate * (train_example.y - pr) * train_example.x

        return self.beta

    def inspect(self, vocab, limit=10):
        """
        A function to find the top features.
        """
        ind_sort = self.beta.argsort()
        indBest = ind_sort[-limit:]
        indWorst = ind_sort[:limit]
        print("Best 10 features based on weight: ")
        print(np.array(vocab)[indBest])
        print("Worst 10 features based on weight: ")
        print(np.array(vocab)[indWorst])

def read_dataset(filename, vocab):
    """
    Reads in a text dataset with a given vocabulary

    filename -- json lines file of the dataset 
    """

    # You should not need to modify this function
    
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    dataset = []
    with open(filename) as infile:
        for line in infile:
            ex = Example(json.loads(line), vocab)
            dataset.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(dataset)

    return dataset

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--vocab", help="Vocabulary of all features",
                           type=str, default="data/small_guess.vocab")
    argparser.add_argument("--train", help="Training set",
                           type=str, default="data/small_guess.buzztrain.jsonl", required=False)
    argparser.add_argument("--test", help="Test set",
                           type=str, default="data/small_guess.buzzdev.jsonl", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()

    with open(args.vocab, 'r') as infile:
        vocab = [x.strip() for x in infile]
    print("Loaded %i items from vocab %s" % (len(vocab), args.vocab))
        
    train = read_dataset(args.train, vocab=vocab)
    test = read_dataset(args.test, vocab=vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.step)
    
    # train_accs = []
    # h_accs = []
    # train_lps = []
    # h_lps = []

    # Iterations
    update_number = 0
    #lr.inspect(vocab)
    for pp in range(args.passes):
        for ii in train:
            update_number += 1
            lr.sg_update(ii)

            if update_number % 100 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)

                #train_accs.append(train_acc)
                #h_accs.append(ho_acc)
                #train_lps.append(train_lp)
                #h_lps.append(ho_lp)

                print("Update %i\tTProb %f\tHProb %f\tTAcc %f\tHAcc %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))
        #lr.inspect(vocab)

    # For getting my plot!
    # X = list(range(0, len(train_accs) * 100, 100))
    # plt.figure(figsize=(15, 10))

    # plt.subplot(3, 1, 1)
    # plt.plot(X, train_accs, label="Training Acc")
    # plt.plot(X, h_accs, label="Holdout Acc")
    # plt.legend()
    # plt.title("Accuracy")

    # plt.subplot(3, 1, 2)
    # plt.plot(X, train_lps)
    # plt.title("Train Log Prob")

    # plt.subplot(3, 1, 3)
    # plt.plot(X, h_lps)
    # plt.title("Holdout Log Prob")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # plt.xlabel("Number of Updates")
    # plt.ylabel("Accuracy %")
    # #plt.show()
    # plt.savefig("sgd/filename", transparent=False)