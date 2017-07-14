
#Spambase dataset with a neural network is 11 hidden neurons in a single hidden layer, and a momentum alpha of 0.1.


import sys, os
import re, string
import math
import numpy as np
from sklearn import preprocessing


def derivative(x):
    return x * (1. - x)
    #return 1. * (x > 0)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))  # Sigmoid
    #return np.tanh(x)              # Tanh
    #return np.maximum(x, x*0.01)  # Leaky ReLU
    #return x * (x > 0)

def softmax(x):
    e = np.exp(x - np.max(x)) # stop overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([ np.sum(e, axis=1) ]).T

def writeFromFile(filepath):

    X = []
    y = []

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

                file_content = fp.read()
                file_content = re.split(r"[^0-9A-Za-z\'_]+", file_content) # maybe put back '-'
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
                n_words = float(len(re.findall(r"[0-9A-Za-z&\'_]+", file_content)))

                input_list = []

                # 1. Total no of characters (C)
                input_list += [ len(file_content) ]

                input_list += [ sum([ 1 for char in file_content if char.isupper() ]) / n_words ]

                # 2. Total no of alpha chars / C Ratio of alpha chars
                num = sum([ 1 for char in file_content if char.isalpha() ]) / n_chars
                input_list += [ math.log(num) if num != 0 else 0.0 ]

                # 3. Total no of digit chars / C
                num = sum([ 1 for char in file_content if char.isdigit() ]) / n_chars
                input_list += [ math.log(num) if num != 0 else 0.0 ]

                # 4. Total no of whitespace chars / C
                num  = sum([ 1 for char in file_content if char.isspace() ]) / n_chars
                input_list += [ math.log(num) if num != 0 else 0.0 ]

                # 5. Frequency of special chars (10 chars: *,_,+,=,%,$,@,\,/)
                input_list += [ file_content.count(char) for char in "*_+=%$@\/&" ]

                # 6. Frequency of punctuation 18 punctuation chars: . , ; ? ! : ( ) - " < > [ ] { }
                input_list += [ file_content.count(char) for char in ".,;?!:()-<>[]{}" ]

                # 7. Frequency of each letter / C (36 letters of the kayboard : A-Z, 0-9)
                for char in string.ascii_letters:
                    input_list += [ file_content.count(char) / n_chars ]
                for num in range(0, 9):
                    input_list += [ file_content.count(str(num)) / n_chars ]

                # 8. Total no of words (M)
                input_list += [ n_words ]

                # 9. Total no of chars in words / C
                input_list += [ math.log( len(re.findall(r"[0-9A-Za-z&\'_]", file_content)) / float(len(file_content)) ) ]


                sentences = re.split(r"[.?!]", file_content)
                n_sentences = float(len(sentences))


                # 10. Avg. sentence length in chars
                input_list += [ (sum([ len(sentence) for sentence in sentences ]) / n_sentences ) ]

                # 11. Avg. sentence length in words
                input_list += [ ( sum([ len(sentence.split()) for sentence in sentences ]) / n_sentences ) ]


                file_content = re.split(r"[^0-9A-Za-z&\'_]+", file_content.lower()) # maybe put back '-'


                # 12. Average word length
                input_list += [ sum([ len(word) for word in file_content ]) / n_words ]

                # 13. Total no of short words / M Two letters or less
                input_list += [ sum([ 1 for word in file_content if len(word) <= 2 ]) / n_words ]

                # 14. Word length freq. distribution / M Ratio of words of length n, n between 1 and 15
                word_set = list(set(file_content))

                for n in range(1,16):
                    input_list += [ sum([ 1 for word in word_set if file_content.count(word) == n ]) / n_words ]
                freq_count = input_list[-15:]
                freq_count = [ n * n_words for n in freq_count ]


                # 15. Type Token Ratio No. Of unique Words / M
                input_list += [ len(word_set) / n_words ]

                # 16. Hapax Legomena Freq. of once-occurring words
                n_once_occurring_words = sum([ 1 for word in word_set if file_content.count(word) == 1 ])
                input_list += [ n_once_occurring_words ]

                # 17. Hapax Dislegomena Freq. of twice-occurring words
                n_twice_occurring_words = sum([ 1 for word in word_set if file_content.count(word) == 2 ])
                input_list += [ n_twice_occurring_words ]

                # 18. Simpson's D measure
                input_list += [ 1 - sum([ file_content.count(word) * (file_content.count(word) - 1) / \
                                         len(word_set) * (len(word_set) - 1) for word in word_set ])  ]

                # 19. Honore's R measure (higher R -> richer vocab)
                #print n_once_occurring_words
                #print len(word_set)
                #print f
                #input_list += [ 100 * math.log(n_words) / (1 - (n_once_occurring_words / len(word_set) )) ]

                # 20. Sichel's S measure
                input_list += [ n_twice_occurring_words / len(word_set) ]

                # 21. Brunet's W measure
                #input_list += [ n_words ** (len(word_set) - 0.17) ]

                # 22. Yule's K measure
                m = sum([ (c+1)**2 * f for c, f in enumerate(freq_count) ])
                input_list += [ 10000 * (m - n_words) / n_words**2 ]


                """
                input_list += [ 10 if word[0] in file_content else 0 for word in most_common_words["ham"] ]
                input_list += [ 10 if word[0] in file_content else 0 for word in most_common_words["spam"] ]
                """

                X.append(input_list)
                y.append(1 if os.path.split(roots)[-1] == "spam" else 0)

                #print X[-1], y[-1]
                #n = raw_input("->")

    X = np.array(X)
    #X = preprocessing.scale(X) # feature scaling
    y = np.array([y]).T
    return X, y


# ======================================================================================
# Program Start


print "\nLoading files...\n"


#if sys.argv[1].lower() == "train":
X_train, y_train = writeFromFile("./data/train/Enron/enron1")



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

print "Done.\n"

dim1 = len(X_train[0])
dim2 = 4

learning_rate = 0.01
bias = 0.0

np.random.seed(1)

weight_0 = 2 * np.random.random((dim1, dim2)) - 1
weight_1 = 2 * np.random.random((dim2, 1)) - 1


print "Training..."


for i in xrange(45000):

    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0, weight_0))
    layer_2 = sigmoid(np.dot(layer_1, weight_1))

    layer_2_error = y_train - layer_2
    layer_2_delta = layer_2_error * derivative(layer_2)

    layer_1_error = layer_2_delta.dot(weight_1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)

    if i % 1000 == 0:
        print i
        print "\n\n"
        print "Error: " + str(np.mean(np.abs(layer_2_error)))
    """
        print "Weight 0 Avg: " + str(np.mean(weight_0))
        print "Weight 1 Avg: " + str(np.mean(weight_1))
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

    weight_1 += layer_1.T.dot(layer_2_delta) * learning_rate
    weight_0 += layer_0.T.dot(layer_1_delta) * learning_rate


print "Done.\n"

print "Weights 0: \n", weight_0, "\n"
print "Weights 1: \n", weight_1, "\n"

print "Error: " + str(np.mean(np.abs(layer_2_error)))
X_test, y_test = writeFromFile("./data/test/enron6")

#np.savetxt("spam_model_nn.txt", weight_1, delimiter=",")


#elif sys.argv[1].lower() == "test":

# evaluation on the testing data
matrix = [[0, 0], \
          [0, 0]]

for i in xrange(len(X_test)):

    layer_0 = X_test[i]
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

