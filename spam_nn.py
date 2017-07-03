
#Spambase dataset with a neural network is 11 hidden neurons in a single hidden layer, and a momentum alpha of 0.1.


import os
import re, string
import numpy as np
from sklearn import preprocessing


def derivative(x):
    return x * (1.0 - x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def writeFromFile(filepath):

    X, y = [], []

    """
    classes = ["ham", "spam"]

    bags_of_words = { "ham" : {}, "spam" : {} }
    most_common_words = { "ham" : [], "spam" : [] }

    sW = open("./stopwords.txt", "r")
    stop_words = [ word for word in sW.read().split() ]
    sW.close()

    for (roots, dirs, files) in os.walk(filepath):

        c = os.path.split(roots)[-1]

        if len(dirs) != 0:
            continue

        for f in files:
            with open(roots + "/" + f) as fp:

                file_contents = fp.read()
                file_contents = re.sub("[\[\]<>\{\}\(\)-.,#:;?!$*\"+_=|~`\\/%&\'\"0-9@]+", " ", file_contents, 0).lower().split()
                file_contents = [ word for word in file_contents if not word in stop_words ]

                for word in file_contents:
                    try:
                        bags_of_words[c][word] += 1
                    except:
                        bags_of_words[c][word] = 0

    for c in classes:
        most_common_words[c] = sorted(bags_of_words[c].iteritems(), key=lambda (k,v): (v,k))[-10:]
    """

    for (roots, dirs, files) in os.walk(filepath):

        if len(dirs):
            continue

        for f in files:
            with open(roots + "/" + f) as fp:

                file_content = fp.read()

                input_list = []

                # Total no of characters (C)
                input_list += [ len(file_content) ]

                # Total no of alpha chars / C Ratio of alpha chars
                #input_list += [ sum([ 1 for char in file_content if char.isalpha() ]) / float(len(file_content)) ]

                # Total no of digit chars / C
                #input_list += [ sum([ 1 for char in file_content if char.isdigit() ]) / float(len(file_content)) ]

                # Total no of whitespace chars / C
                #input_list += [ sum([ 1 for char in file_content if char.isspace() ]) / float(len(file_content)) ]

                # Frequency of each letter / C (36 letters of the kayboard : A-Z, 0-9)
                #for char in string.ascii_lowercase:
                #    input_list += [ file_content.lower().count(char) / float(len(file_content)) ]

                #for num in range(0,9):
                #    input_list += [ file_content.count(str(num)) / float(len(file_content)) ]

                # Frequency of special chars (10 chars: *,_,+,=,%,$,@,\,/)
                for char in "*_+=%$@\/":
                    input_list += [ float(file_content.count(char)) ]

                # Frequency of punctuation 18 punctuation chars: . , ; ? ! : ( ) - " < > [ ] { }
                for char in ".,;?!:()-<>[]{}":
                    input_list += [ file_content.count(char) ]

                # Total no of words (M)
                input_list += [ len(re.findall(r"\w+", file_content)) ]

                #Total no of chars in words / C
                #input_list += [ len(re.findall(r"\w", file_content)) / float(len(file_content)) ]

                # Avg. sentence length in chars
                sentences = re.split(r"[.?!]", file_content)
                #input_list += [ sum([ len(sentence) for sentence in sentences ]) / float(len(sentences)) ]

                # Avg. sentence length in words
                #input_list += [ sum([ len(sentence.split()) for sentence in sentences ]) / float(len(sentences)) ]

                file_content = re.split(r"[^0-9A-Za-z\'_]+", file_content) # maybe put back '-'
                #print file_content
                #input("")

                # Average word length
                #input_list += [ sum([ len(word) for word in file_content ]) / float(len(file_content)) ]

                # Total no of short words/M Two letters or less
                input_list += [ sum([ 1 for word in file_content if len(word) <= 2 ]) ]

                # Word length freq. distribution/M Ratio of words of length n, n between 1 and 15
                #for n in range(1,15):
                #    input_list += [ sum([ 1 for word in file_content if len(word) == n ]) / float(len(file_content)) ]

                # Type Token Ratio No. Of unique Words / M

                word_set = set(file_content)

                #input_list += [ len(word_set) / float(len(file_content)) ]

                # Hapax Legomena Freq. of once-occurring words
                input_list += [ sum([ 1 for word in word_set if file_content.count(word) == 1 ]) ]

                # Hapax Dislegomena Freq. of twice-occurring words
                input_list += [ sum([ 1 for word in word_set if file_content.count(word) == 2 ]) ]

                #print input_list
                #input("")

                X.append(input_list)
                y.append(1 if os.path.split(roots)[-1] == "spam" else 0)

    X = np.array(X)
    #X = preprocessing.scale(X) # feature scaling
    y = np.array([y]).T
    return (X, y)

"""
input_list += [ 1 if word[0] in file_content else 0 for word in most_common_words["ham"] ]
input_list += [ 1 if word[0] in file_content else 0 for word in most_common_words["spam"] ]
input_list += [ sum([ 1 for word in file_content if word[0].isupper() ]) / float(len(file_content)) ]

Yule's K measure
Simpson's D measure
Sichel's S measure
Brunet's W measure
Honore's R measrue

color, font, size
"""

# ===================================================
# Program Start


print "\nLoading files...\n"

X_train, y_train = writeFromFile("./data/train/Enron/enron1")
X_test, y_test = writeFromFile("./data/train/Enron/enron3")


"""

X_train = np.array([[3, 1, 1, 0, 1],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0],
                    [5, 1, 0, 0, 0]])

y_train = np.array([[1],
                    [0],
                    [1],
                    [0],
                    [1]])
"""


print "Done.\n"

dim1 = len(X_train[0])
dim2 = 4

learning_rate = 1.0
bias = 0.0

#np.random.seed(1)

weight_0 = 2 * np.random.random((dim1, dim2)) - 1
weight_1 = 2 * np.random.random((dim2, 1)) - 1

print "Training..."

for i in xrange(35000):
    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0, weight_0))
    layer_2 = sigmoid(np.dot(layer_1, weight_1))

    layer_2_error = y_train - layer_2
    layer_2_delta = layer_2_error * derivative(layer_2)

    layer_1_error = layer_2_delta.dot(weight_1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)

    if i % 1000 == 0:
        print "Error: " + str(np.mean(np.abs(layer_2_error)))
        print "Weight 0 Avg: " + str(np.mean(weight_0))
        print "Weight 1 Avg: " + str(np.mean(weight_1))
        print "\n"

    weight_1 += layer_1.T.dot(layer_2_delta)
    weight_0 += layer_0.T.dot(layer_1_delta)

    """
    if i % 1000 == 0:
        print "-"
        print layer_1.T
        print layer_2_delta

        print layer_1.T.dot(layer_2_delta)
        print layer_1.T.dot(layer_2_delta)
    """

print "Done.\n"

print "Weights 0: \n", weight_0, "\n"
print "Weights 1: \n", weight_1, "\n"


# evaluation on the testing data
matrix = [[0, 0], \
          [0, 0]]

for i in xrange(len(X_test)):

    layer_0 = np.array(X_test[i])
    layer_1 = sigmoid(np.dot(layer_0, weight_0))
    layer_2 = sigmoid(np.dot(layer_1, weight_1))

    #print X_test[i], "\n", layer_1, "\n", layer_2[0], y_test[i], "\n"

    layer_2[0] = 1 if layer_2[0] > 0.50 else 0

    matrix[ int(layer_2[0]) ][ int(y_test[i]) ] += 1

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
n_correct = sum([ matrix[i][i] for i in range(len(matrix[0])) ])

print "Total: ", len(X_test)
print "Correct: ", n_correct
print "Accuracy: ", n_correct / float(len(X_test)) * 100.0, "\n"


