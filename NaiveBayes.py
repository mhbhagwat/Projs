__author__ = 'mrunmayee'

# This program performs sentiment analysis for given files. Here the files considered are movie reviews and
# the program classifies them into positive and negative reviews. The approach used is Naive Bayes algorithm.
# The input to this program is a directory path which contains folders namely "pos" and "neg" which further contain
# text files (movie reviews).
# The data set of positive and negative files is divided into three parts and the algorithm performs three iterations
# in which one part from these three is the test set and remaining two form the training set.
# The accuracy for every iteration and the average accuracy for three iterations is calculated based on the number
# of files classified correctly.

import argparse
import glob
import re
import random
import collections
import math
import numpy as np

stopWords = ['able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'among', 'and', 'any', 'are', 'because', 'been', 'but', 'can',
             'cannot', 'could', 'dear', 'did', 'does', 'either', 'else',
             'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has',
             'have', 'her', 'here' 'hers', 'him', 'his', 'how', 'however',
             'into', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'might', 'most', 'must', 'neither', 'nor', 'not', 'off', 'often',
             'only', 'other', 'our', 'own', 'put', 'rather', 'said', 'say',
             'says', 'she', 'should', 'since', 'some', 'such', 'than', 'that',
             'the', 'their', 'them','then', 'there', 'these', 'they', 'this',
             'tis', 'too', 'twas', 'wants', 'was', 'were', 'what', 'when',
             'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
             'would', 'yet', 'you', 'your', 'www', 'http', 'women', 'males',
             'each', 'done', 'see', 'before', 'each', 'irs', 'ira', 'hal', 'ham']

count_sw = collections.Counter(stopWords)

def parse_argument():
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def get_words(f):
    all_words = []
    read_file = open(f, "r")
    for line in read_file:
        line_sub = re.sub("[^\[a-zA-Z]]*", " ", line)      # Substitute all the nos., punctuation marks with a space
        for word in line_sub.split():
            if len(word) > 2 and count_sw[word] == 0:      # Remove stopwords and words with less than two letters
                all_words.append(word.lower())
    return all_words


def read_all(list_files):
    total_words = []                                       # Stores all the words from the given list of files
    for f in list_files:
        value = get_words(f)
        total_words.extend(value)
    return total_words


def classify(directory):
    list_pos = glob.glob(directory + "/" + "pos/*.txt")
    list_neg = glob.glob(directory + "/" + "neg/*.txt")

    random.shuffle(list_pos)
    random.shuffle(list_neg)

    len_pos = len(list_pos) / 3                            # Divide list of positive files in three parts
    pt1 = list_pos[0:len_pos]
    pt2 = list_pos[len_pos: 2 * len_pos]
    pt3 = list_pos[2 * len_pos:]

    len_neg = len(list_neg) / 3                            # Divide list of negative files in three parts
    nt1 = list_neg[0:len_neg]
    nt2 = list_neg[len_neg: 2 * len_neg]
    nt3 = list_neg[2 * len_neg:]

    pos_t1 = pt1                                           # Training and test sets
    pos_t2 = pt2
    pos_test = pt3
    neg_t1 = nt1
    neg_t2 = nt2
    neg_test = nt3

    total_accuracy = 0
    for j in range(0, 3):

        list_test = pos_test + neg_test                    # Shuffle test set
        random.shuffle(list_test)
        len_test_files = len(list_test)

        list_pos_training = pos_t1 + pos_t2                # Combining the two training sets
        list_neg_training = neg_t1 + neg_t2

        all_pos_words = read_all(list_pos_training)
        all_neg_words = read_all(list_neg_training)
        c_pos = collections.Counter(all_pos_words)         # Frequency of words in pos and neg training sets
        c_neg = collections.Counter(all_neg_words)

        total_pos = len(all_pos_words)                     # Length of total positive words
        total_neg = len(all_neg_words)                     # Length of total positive words

        total_words = []
        total_words.extend(list(set(all_pos_words)))
        total_words.extend(list(set(all_neg_words)))
        unq_pos_neg = len(list(set(total_words)))          # Unique positive and negative words

        accurate = 0
        accuracy_pos = 0
        accuracy_neg = 0

        prob_pos = len(pos_test) / float(len_test_files)   # Probability of pos & neg documents in the training set
        prob_neg = len(neg_test) / float(len_test_files)

        den_pos = float(total_pos + unq_pos_neg + 1)             # Calculate denominator for pos & neg conditional
        unk_pos = 1 / den_pos                                    # probabilities
        den_neg = float(total_neg + unq_pos_neg + 1)
        unk_neg = 1 / den_neg

        for i in range(0, len_test_files):

            count = 0
            filename = list_test[i]
            file_class = filename.split("/")[-2]                     # Get the original file class i.e. pos or neg
            words_test = get_words(filename)                         # Get a list of all words in test file
            c_test = collections.Counter(words_test)                 # Frequency of words in test file
            unq_test_words = list(set(words_test))                   # Get a list of unique words in test file

            final_pos = math.log(prob_pos, math.e)                   # Stores final pos & neg conditional probabilities
            final_neg = math.log(prob_neg, math.e)

            xp = np.zeros(len(unq_test_words))
            xn = np.zeros(len(unq_test_words))

            for word in unq_test_words:

                count_test_words = c_test[word]                      # Frequency of 'word' in the test file

                if c_pos[word] != 0:                                 # Calculate positive score for the test file
                    p1 = (c_pos[word] + 1) / den_pos
                else:
                    p1 = unk_pos
                xp[count] = count_test_words * math.log(p1, math.e)

                if c_neg[word] != 0:                                 # Calculate negative score for the test file
                    p2 = (c_neg[word] + 1) / den_neg
                else:
                    p2 = unk_neg
                xn[count] = count_test_words * math.log(p2, math.e)

                count += 1

            final_pos += xp.sum()
            final_neg += xn.sum()

            if final_pos > final_neg:                                # Classify the test file as pos/ neg
                result = "pos"
            else:
                result = "neg"
            if result == file_class:
                if result == "pos":
                    accuracy_pos += 1
                elif result == "neg":
                    accuracy_neg += 1
                accurate += 1

        accuracy = (accurate / float(len_test_files)) * 100          # Find accuracy
        total_accuracy += accuracy

        print "iteration %d:" % (j + 1)                              # Printing output
        print "num_pos_test_docs:%d" % len(pos_test)
        print "num_pos_training_docs:%d" % len(list_pos_training)
        print "num_pos_correct_docs:%d" % accuracy_pos
        print "num_neg_test_docs:%d" % len(neg_test)
        print "num_neg_training_docs:%d" % len(list_neg_training)
        print "num_neg_correct_docs:%d" % accuracy_neg
        print "accuracy:%.2f%s" % (accuracy, "%")

        pos_t1 = pos_t2                                              # Redefine training and test sets
        pos_t2 = pos_test
        pos_test = pt1
        pt1 = pt2
        neg_t1 = neg_t2
        neg_t2 = neg_test
        neg_test = nt1
        nt1 = nt2

    print "ave_accuracy:%.2f%s" % (total_accuracy / (j + 1), "%")
    return


def main():
    args = parse_argument()
    directory = args['d'][0]
    classify(directory)

main()