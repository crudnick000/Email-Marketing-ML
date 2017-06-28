
#Spambase dataset with a neural network is 11 hidden neurons in a single hidden layer, and a momentum alpha of 0.1. 

import re
import sys, os
from sklearn import preprocessing
import numpy as np


def derivative(x):
    return x * (1.0 - x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def writeFromFile(filepath):

    X, y = [], []

    for (roots, dirs, files) in os.walk(filepath):

        if not len(files) > 1:
            continue

        for f in files:
            with open(roots + "/" + f) as fp:

                file_content = fp.read()

                inputList = []

                # # of !
                inputList += [ file_content.count("!") ]

                file_content = file_content.split()

                # % of Capitalized words
                inputList += [ sum([ 1 for word in file_content if word[0].isupper() ]) / float(len(file_content)) ]

                # # of $
                inputList += [ file_content.count("$") ]

            X.append(inputList)
            y.append(os.path.split(roots)[-1])

    X = np.array(X)
    #X = preprocessing.scale(X) # feature scaling
    y = np.array([ 1.0 if i == "spam" else 0.0 for i in y ])

    return X, y

# ===================================================
# Program Start

# Features
#     number of strong punctuators ex: !!!
#     percentage of captial words
#     number of currency values found
#     percentage of colored text

_directory = sys.argv[1]

print "\nLoading files..."

X_train, y_train = writeFromFile("./data/train")
X_test, y_test = writeFromFile("./data/test")

print "Done.\n"

dim1 = len(X_train[0])
dim2 = 4

np.random.seed(1)
weight_0 = 2 * np.random.random((dim1, dim2)) - 1
weight_1 = 2 * np.random.random((dim2, 1)) - 1

print "Training..."

for h in xrange(10):
    for i in xrange(len(X_train)):

        layer_0 = np.array([X_train[i]])
        layer_1 = sigmoid(np.dot(layer_0, weight_0))
        layer_2 = sigmoid(np.dot(layer_1, weight_1))

        layer_2_error = y_train[i] - layer_2
        layer_2_delta = layer_2_error * derivative(layer_2)

        layer_1_error = layer_2_delta.dot(weight_1.T)
        layer_1_delta = layer_1_error * derivative(layer_1)

        weight_1 += layer_1.T.dot(layer_2_delta)
        weight_0 += layer_0.T.dot(layer_1_delta)

print "Done.\n"

print "Weights 0: \n", weight_0, "\n"
print "Weights 1: \n", weight_1, "\n"

# evaluation on the testing data
correct = 0
matrix = [[0, 0], \
          [0, 0]]

for i in xrange(len(X_test)):

    layer_0 = X_test[i]
    layer_1 = sigmoid(np.dot(layer_0, weight_0))
    layer_2 = sigmoid(np.dot(layer_1, weight_1))

    """
    print layer_2[0]
    print y_test[i], "\n"
    """

    layer_2[0] = 1 if layer_2 > 0.50 else 0

    matrix[int(layer_2[0])][int(y_test[i])] += 1

    if layer_2[0] == y_test[i]:
        correct += 1

topics = ["Spam", "Ham"]

print "\n\nConfusion Matrix:"
print " a\p  " + "".join([ " %5s" % topic for topic in topics ])
print "      " + "-------" * len(topics)

for row in range(len(topics)):
    current_row = "%4s |" % topics[row]
    for col in range(len(topics)):
        current_row += " %5s" % int(matrix[row][col],)
    print current_row
print "\n"



# printing the output
print "Total: ", len(X_test)
print "Correct: ", correct
print "Accuracy: ", correct / float(len(X_test)) * 100.0, "\n"


"""
Total no of characters (C)
Total no of alpha chars / C Ratio of alpha chars
Total no of digit chars / C
Total no of whitespace chars/C
Frequency of each letter / C (36 letters of the kayboard : A-Z, 0-9)
Frequency of special chars (10 chars: *,_,+,=,%,$,@,\,/)
Total no of words (M)
Total no of short words/M Two letters or less
Total no of chars in words/C
Average word length
Avg. sentence length in chars
Avg. sentence length in words
Word length freq. distribution/M Ratio of words of length n, n between 1 and 15
Type Token Ratio No. Of unique Words/ M
Hapax Legomena Freq. of once-occurring words
Hapax Dislegomena Freq. of twice-occurring words
Yule's K measure
Simpson's D measure
Sichel's S measure
Brunet's W measure
Honore's R measrue
Frequency of punctuation 18 punctuation chars: . , ; ? ! : ( ) - " < > [ ] { }
color, font, size
"""
