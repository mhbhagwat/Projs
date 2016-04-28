__author__ = 'mrunmayee'


from collections import Counter
import re
import math

########################################################################
# Use the following code to run on AWS cluster
########################################################################
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""

sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

pos_files = sc.wholeTextFiles("s3n://aml-hw2/trainpos/*.txt").repartition(16)
neg_files = sc.wholeTextFiles("s3n://aml-hw2/trainneg/*.txt").repartition(16)
test_pos_files = sc.wholeTextFiles("s3n://aml-hw2/testpos/*.txt").repartition(16)
test_neg_files = sc.wholeTextFiles("s3n://aml-hw2/testneg/*.txt").repartition(16)
########################################################################

########################################################################
# Uncomment the following and change the path when running on local machine
########################################################################
# path = "/Users/mrunmayee/AdvancedML/Asg2/aclImdb/"
# pos_files = sc.wholeTextFiles(path + "train/pos/*.txt").repartition(16)
# neg_files = sc.wholeTextFiles(path + "train/neg/*.txt").repartition(16)
# test_pos_files = sc.wholeTextFiles(path + "test/pos/*.txt").repartition(16)
# test_neg_files = sc.wholeTextFiles(path + "test/neg/*.txt").repartition(16)

stop_words = ['able', 'about', 'across', 'after', 'all', 'almost', 'also',
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
             'each', 'done', 'see', 'before', 'each', 'irs', 'ira', 'hal', 'ham', 'isn']

count_sw = Counter(stop_words)
def remove_stop_small(words):
    l = []
    for wr in words:
        w = wr.lower()
        if count_sw[w] == 0 and len(w) > 2:
            l.append(w)
    return l

def getwords(text):
    line_sub = re.sub("[ ]+", " ", re.sub("[^\[a-zA-Z]]*", " ", text)).split(" ")
    return line_sub


# pos_files is an RDD of (file name, text) in the positive train set
# all_pos_words is an RDD - list of all words in the positive train set
# c_pos is a list - counter for words in the positive train set
# total_pos - count of all words in the positive train set
all_pos_words = pos_files.map(lambda x: getwords(x[1])).flatMap(remove_stop_small)
total_pos = all_pos_words.count()
c_pos = all_pos_words.countByValue().items()
c_pos = dict(c_pos)

all_neg_words = neg_files.map(lambda x: getwords(x[1])).flatMap(remove_stop_small)
total_neg = all_neg_words.count()
c_neg = all_neg_words.countByValue().items()
c_neg = dict(c_neg)


# unq_pos_neg - count of unique words in positive and negative train set
# total_words - list of unique words from positive and negative train set
# but some repeated words
total_words = []
total_words.extend(list(set(all_pos_words.collect())))
total_words.extend(list(set(all_neg_words.collect())))
unq_pos_neg = len(list(set(total_words)))

# Probability of positive and negative train sets
total_no_files = (pos_files.count() + neg_files.count()) * 1.0
prob_pos = pos_files.count() / total_no_files
prob_neg = neg_files.count() / total_no_files

den_pos = float(total_pos + unq_pos_neg + 1)               # Calculate denominator for pos & neg conditional
unk_pos = 1 / den_pos                                      # probabilities
den_neg = float(total_neg + unq_pos_neg + 1)
unk_neg = 1 / den_neg
final_pos = math.log(prob_pos, math.e)                     # Stores final pos & neg conditional probabilities
final_neg = math.log(prob_neg, math.e)


def calc_prob(x, c_pos, c_neg, den_pos, unk_pos, den_neg, unk_neg, final_pos, final_neg):
    for word in x:
        count_test_words = x[word]                            # Frequency of 'word' in the test file
        if word in c_pos:                                     # Calculate positive score for the test file
            p1 = (c_pos[word] + 1) / den_pos
        else:
            p1 = unk_pos
        final_pos += (count_test_words * math.log(p1, math.e))

        if word in c_neg:                                     # Calculate negative score for the test file
            p2 = (c_neg[word] + 1) / den_neg
        else:
            p2 = unk_neg
        final_neg += count_test_words * math.log(p2, math.e)
        if final_pos > final_neg:
            res = "pos"
        else:
            res = "neg"
    return res

# Read the test files from 'pos' and 'neg' folders and merge them in 'test_files'
pos_words = test_pos_files.map(lambda x: getwords(x[1])).map(remove_stop_small)
neg_words = test_neg_files.map(lambda x: getwords(x[1])).map(remove_stop_small)

pos_predict = pos_words.map(lambda x: Counter(x)).map(lambda x: calc_prob(x, c_pos, c_neg, den_pos, unk_pos, den_neg, unk_neg, final_pos, final_neg)).filter(lambda x: x == "pos")
neg_predict = neg_words.map(lambda x: Counter(x)).map(lambda x: calc_prob(x, c_pos, c_neg, den_pos, unk_pos, den_neg, unk_neg, final_pos, final_neg)).filter(lambda x: x == "neg" )


cor_total = pos_predict.count() + neg_predict.count()
total = test_pos_files.count() + test_neg_files.count()


print "Number of positive files classified correctly = ", pos_predict.count()
print "Number of negative files classified correctly = ", neg_predict.count()
print "Total number of test data set files = ", total
print "Accuracy = ", cor_total * 100.0 / total