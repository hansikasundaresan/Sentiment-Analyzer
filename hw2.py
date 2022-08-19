import re
import sys

import nltk
import numpy
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    f = open(corpus_path,"r", encoding='latin-1')
    sentences = f.read().split("\n")
    single = []
    answer = []
    for i in sentences:
        punctuation = []
        single = i.split(" ")
        if single == ['']:
            return answer
        punctuation = single[len(single)-1].split("\t")
        single[len(single)-1] = punctuation[0]
        temp_tuple = (single , int(punctuation[1]))
        answer.append(temp_tuple)
    return answer


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words or "n't" in word:
        return True
    return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    sentence = nltk.pos_tag(snippet)
    temp = [""]*len(sentence)
    for i in range(len(sentence)):
        if is_negation(sentence[i][0]):
            if i+1 != len(sentence):
                if sentence[i+1][0] == "only":
                    continue
                i = i+1
                while i != len(sentence) and sentence[i][0] not in negation_enders and sentence[i][0] not in sentence_enders and sentence[i][1] != "JJR" and sentence[i][1] != "RBR":
                    temp[i] = "NOT_"
                    i = i+1
                i = i-1
    answer = []
    for k,j in zip(sentence,temp):
        answer.append(j+k[0])
    return answer
                    


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    index = 0
    dict = {}
    for i in corpus:
        for j in i[0]:
           if j not in dict:
            dict[j] = index
            index = index+1
    return dict
    

# Converts a snippet into a feature vector
# snippet is a list of strings
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    zeros = numpy.zeros(len(feature_dict), dtype=int)
    for i in snippet:
        if i in feature_dict:
            zeros[int(feature_dict[i])] = zeros[int(feature_dict[i])]+1
    return zeros


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    X = numpy.empty([len(corpus),len(feature_dict)])
    Y = numpy.empty(len(corpus))
    for i in range(len(corpus)):
        feature = vectorize_snippet(corpus[i][0], feature_dict)
        X[i] = feature
        Y[i] = corpus[i][1]
    return (X,Y)
    


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    transpose = numpy.transpose(X)
    for i in range(len(transpose)):
        max = numpy.max(transpose[i])
        min = numpy.min(transpose[i])
        if min == max:
            continue
        for j in range(len(transpose[i])):
            X[j][i] = (X[j][i]-min)/(max-min)



# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    sentences = load_corpus(corpus_path)
    corpus =[]
    for i in sentences:
        corpus.append((tag_negation(i[0]), i[1]))
    dict = get_feature_dictionary(corpus)
    X = vectorize_corpus(corpus, dict)
    normalize(X[0])
    model = LogisticRegression().fit(X[0],X[1])
    return (model, dict)

# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp =0
    fp = 0
    fn = 0
    for i,j in zip(Y_pred, Y_test):
        if i == 1 and j == 1:
            tp+=1
        elif i==1 and j==0:
            fp+=1
        elif i==0 and j==1:
            fn+=1
    p = float(tp)/(tp+fp)
    r = float(tp)/(tp+fn)
    f = 2 * (p*r)/(p+r)
    return (p,r,f)


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    sentences = load_corpus(corpus_path)
    corpus =[]
    for i in sentences:
        corpus.append((tag_negation(i[0]), i[1]))
    X = vectorize_corpus(corpus, feature_dict)
    test = model.predict(X[0])
    return evaluate_predictions(X[1], test)

# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    co = logreg_model.coef_
    list_tuples = []
    for i in range(len(co)):
        list_tuples.append([i, co[i]])
    list_tuples.sort(key=lambda row: abs(row[1]))
    key_list = list(feature_dict.keys())
    val_list = list(feature_dict.values())
    ans = []
    for i in range(len(list_tuples)):
        position = val_list.index(list_tuples[i][0])
        unigram = key_list[position]
        list_tuples[i][0] = unigram
        if len(ans)<k:
            ans.append(tuple(list_tuples[i]))
    return ans

def main(args):
    model, feature_dict = train('train.txt')
    
    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
