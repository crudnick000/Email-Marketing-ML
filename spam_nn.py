
#Spambase dataset with a neural network is 11 hidden neurons in a single hidden layer, and a momentum alpha of 0.1.


import csv
import sys
import os
import re, string
import math
import numpy as np
from sklearn import preprocessing

class NeuralNetwork():
    def __init__(self):
        pass

def tanh(x, derivative=False):
    if derivative == True:
        return 1 - (x ** 2)
    return np.tanh(x)

def sigmoid(x, derivative=False):
    if derivative == True:
        return x * (1 - x)
    return 1. / (1 + np.exp(-x))

def relu(x, derivative=False):
    if derivative == True:
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    """
    for i in range(len(x)):
        for k in range(len(x[i])):
            if x[i][k] > 0:
                pass
            else:
                x[i][k] *= 0.001
    return x
    """
    return np.maximum(0, x)

def arctan(x, derivative=False):
    if derivative == True:
        return (np.cos(x) ** 2)
    return np.arctan(x)


def writeFromFile(filepath):

    X = []
    y = []

    bags_of_words = { "ham" : {}, "spam" : {} }
    most_common_words = { "ham" : [], "spam" : [] }

    sw = open("./stopwords.txt", "r")
    stop_words = [ word for word in sw.read().split() ]
    sw.close()

    for (roots, dirs, files) in os.walk(filepath):

        c = os.path.split(roots)[-1]

        if len(dirs) != 0:
            continue

        for f in files:
            with open(roots + "/" + f) as fp:

                file_content = fp.read()
                file_content = re.split(r"[^0-9a-za-z\'_]+", file_content) # maybe put back '-'
                file_content = [ word for word in file_content if not word in stop_words ]

                for word in file_content:
                    try:
                        bags_of_words[c][word] += 1
                    except:
                        bags_of_words[c][word] = 0

    classes = ["spam", "ham"]
    for c in classes:
        most_common_words[c] = sorted(bags_of_words[c].iteritems(), key=lambda (k,v): (v,k))[-10:]

    n_files = sum([ len(files) for (roots, dirs, files) in os.walk(filepath) ])
    file_count = 1

    for (roots, dirs, files) in os.walk(filepath):

        if len(dirs) != 0:
            continue

        for f in files:
            sys.stdout.flush()
            sys.stdout.write("Files read (%s): %d out of %d         \r" % (roots, file_count, n_files))

            file_count += 1
            with open(roots + "/" + f) as fp:

                file_content = fp.read()
                n_chars = float(len(file_content))
                n_words = float(len(re.findall(r"[0-9A-Za-z&\']+", file_content))) # use _ ?

                input_list = []

                # 1. Total no of characters (C)
                #input_list += [ len(file_content) ]

                # 2. Total # of uppercase chars / # words
                input_list += [ sum( 1 for char in file_content if char.isupper() ) / n_words ]

                # 3. Total no of alpha chars / C Ratio of alpha chars
                num = sum( 1 for char in file_content if char.isalpha() ) / n_chars
                #input_list += [ num ]

                # 4. Total no of digit chars / C
                num = sum( 1 for char in file_content if char.isdigit() ) / n_chars
                #input_list += [ num ]

                # 5. Total no of whitespace chars / C
                num  = sum( 1 for char in file_content if char.isspace() ) / n_chars
                #input_list += [ num ]

                # 6. Frequency of special chars (10 chars: *,_,+,=,%,$,@,\,/)
                input_list += [ file_content.count(char) for char in "%$@\/&" ] # *_+=

                # 7. Frequency of punctuation 18 punctuation chars: . , ; ? ! : ( ) - " < > [ ] { }
                input_list += [ file_content.count(char) for char in "?!" ] # .,;:()-<>[]{}

                # 8. Frequency of each letter / C (36 letters of the kayboard : A-Z, 0-9)
                #for char in string.ascii_letters:
                #    input_list += [ file_content.count(char) / n_chars ]

                #for num in range(0, 9):
                #    input_list += [ file_content.count(str(num)) / n_chars ]

                # 9. Total no of words (M)
                #input_list += [ n_words ]

                # 10. Total no of chars in words / C
                #input_list += [ len(re.findall(r"[0-9A-Za-z&\']", file_content)) / n_chars ] # use _ ?


                sentences = re.split(r"[.?!]", file_content)
                n_sentences = float(len(sentences))


                # 11. Avg. sentence length in chars
                input_list += [ sum( len(sentence) for sentence in sentences ) - 1 / n_sentences ]

                # 12. Avg. sentence length in words
                #input_list += [ sum( len(sentence.split()) for sentence in sentences ) / n_sentences ]


                file_tokens = re.split(r"[^0-9A-Za-z&\']+", file_content.lower()) # Use - and _ ?
                #print file_tokens
                #n=raw_input("->")

                # 13. Average word length
                #input_list += [ sum( len(word) for word in file_tokens ) / n_words ]

                # 14. Total no of short words (Two letters or less) / M
                input_list += [ sum( 1 for word in file_tokens if len(word) <= 2 ) / n_words ]

                # 15. Word length freq. distribution / M Ratio of words of length n, n between 1 and 15
                word_set = list(set(file_tokens))

                for n in range(1,16):
                    input_list += [ sum([ 1 for word in word_set if file_tokens.count(word) == n ]) / n_words ]
                freq_count = input_list[-15:]
                freq_count = [ n * n_words for n in freq_count ]


                # 16. Type Token Ratio No. Of unique Words / M
                input_list += [ len(word_set) / n_words ]

                # 17. Hapax Legomena Freq. of once-occurring words
                n_once_occurring_words = sum([ 1 for word in word_set if file_tokens.count(word) == 1 ])
                input_list += [ n_once_occurring_words ]

                # 18. Hapax Dislegomena Freq. of twice-occurring words
                n_twice_occurring_words = sum([ 1 for word in word_set if file_tokens.count(word) == 2 ])
                input_list += [ n_twice_occurring_words ]

                # 19. Simpson's D measure
                input_list += [ 1 - sum([ file_tokens.count(word) * (file_tokens.count(word) - 1) / \
                                         len(word_set) * (len(word_set) - 1) for word in word_set ])  ]

                # 20. Honore's R measure (higher R -> richer vocab)
                factor = 0.999 if n_once_occurring_words == len(word_set) else n_once_occurring_words / len(word_set)
                input_list += [ 100 * math.log(n_words) / (1 - (factor) ) ]

                # 21. Sichel's S measure
                input_list += [ n_twice_occurring_words / len(word_set) ]

                # 22. Brunet's W measure
                input_list += [ math.pow(n_words, math.pow(len(word_set), -0.165)) ]

                # 23. Yule's K measure
                m = sum( math.pow(c + 1, 2) * f for c, f in enumerate(freq_count) )
                input_list += [ 10000 * (m - n_words) / math.pow(n_words, 2) ]


                input_list += [ 1 if file_tokens.count(word[0]) else 0 for word in most_common_words["ham"] ]
                input_list += [ 1 if word[0] in file_tokens else 0 for word in most_common_words["spam"] ]

                X.append(input_list)
                y.append(1 if os.path.split(roots)[-1] == "spam" else 0)

                #print f
                #print X[0], y[0]
                #n = raw_input("->")

    X = np.array(X)
    X = preprocessing.scale(X) # feature scaling
    y = np.array([y]).T
    return X, y

def predict(x, weights):
    print weights[0].shape
    layer = tanh(np.dot(x, weights[0]))
    for i in range(1, len(weights)):
        layer = tanh(np.dot(layer, weights[i]))
    return 1 if layer > 0.5 else 0

def writeModelToFile(self):

    with open("nn_modelFile.csv", "w") as fp:
        writer = csv.writer(fp)
        for weight in weights:
            writer.writerow(weight.shape)
            for row in weight:
                writer.writerow(row)

def readModelToFile(self):

    weights = []
    with open("nn_modelFile.csv", "r") as fp:
        reader = csv.reader(fp)
        dim = [ int(i) for i in reader.next() ]
        while len(dim) != 0:
            weight = []
            for i in range(dim[0]):
                weight += [ float(i) for i in reader.next() ]
            weights += [ np.array([weight]).reshape(dim[0], dim[1]) ]
            try:
                dim = [ int(i) for i in reader.next() ]
            except StopIteration:
                break
    return weights

# ==============================================================================
# Program Start


learning_rate = 0.10
bias = 0.0

weights = []

if sys.argv[3] == "train":

    print "\nLoading files...\n"

    X_train, y_train = writeFromFile("./data/train/Enron/enron1")

    print "Done.\n"

    dim1 = len(X_train[0])
    dim2 = 4

    np.random.seed(1)

    weights += [ 2 * np.random.random((dim1, dim2)) - 1 ]
    weights += [ 2 * np.random.random((dim2, 1)) - 1 ]

    print "Training..."


    for i in xrange(55000):

        layer_0 = X_train
        layer_1 = tanh(np.dot(layer_0, weights[0]))
        layer_2 = tanh(np.dot(layer_1, weights[1]))

        layer_2_error = y_train - layer_2
        layer_2_delta = layer_2_error * tanh(layer_2, derivative=True)

        layer_1_error = layer_2_delta.dot(weights[1].T)
        layer_1_delta = layer_1_error * tanh(layer_1, derivative=True)

        if i % 1000 == 0:
            print i
            print "\n"
            print "Error: " + str(np.mean(np.abs(layer_2_error)))
        """
            print "Weight 0 Avg: " + str(np.mean(weights[0]))
            print "Weight 1 Avg: " + str(np.mean(weights[1]))
            print "\n"
            print "layer_2_error: " + str(layer_2_error)
            print "layer_2_der: " + str(derivative(layer_2))
            print "layer_2_delta: " + str(layer_2_delta)
            print "\n"
            print "layer_1_error: " + str(layer_1_error)
            print "layer_1_der: " + str(derivative(layer_1))
            print "layer_1_delta: " + str(layer_1_delta)
            n = raw_input("->")
         """

        weights[1] += layer_1.T.dot(layer_2_delta) * learning_rate
        weights[0] += layer_0.T.dot(layer_1_delta) * learning_rate


    print "Done.\n"

    print "Weights 0: \n", weights[0], "\n"
    print "Weights 1: \n", weights[1], "\n"

    print "Error: " + str(np.mean(np.abs(layer_2_error)))

elif sys.argv[3] == "test":

    weights = readModelToFile(weights)
    X_test, y_test = writeFromFile("./data/test/enron6")

    # evaluation on the testing data
    matrix = [[0, 0], \
              [0, 0]]

    for i in xrange(len(X_test)):

        result = predict(X_test[i], [weights[0], weights[1]])
        #print X_test[i], "\n", layer_1, "\n", layer_2[0], y_test[i], "\n"

        matrix[ int(result) ][ int(y_test[i]) ] += 1

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


    """
    X = np.array([[3, 1, 1, 0, 0.6, 1],
                        [0, 0, 1, 0, 0.1, 0],
                        [0, 0, 1, 0, 0.89,0],
                        [0, 1, 1, 1, 0.1111111, 0],
                        [0, 1, 1, 1, 0.942, 0],
                        [1, 0, 0, 0, 0.2135, 0],
                        [5, 1, 0, 0, 0.2042, 0]])

    y = np.array([[1],
                        [0],
                        [0],
                        [1],
                        [1],
                        [0],
                        [1]])

    X_test = np.array([[4, 1, 1, 1, 0.54444, 1],
                       [0, 0, 0, 0, 0.1408, 0]])

    """

else:
    sys.exit("Neural Network Error: Invalid mode.")


