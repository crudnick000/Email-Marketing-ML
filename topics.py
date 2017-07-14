#!/usr/bin/env python

import os
import sys
import math
import re
import time
#import stem

"""
    Thoughts
        use neutral words
        bias toward words nearly guarenteed to be in topic
"""

class BayesNet():
    def __init__(self):
        self.T_given_D = {}
        self.W_given_T = {}


    def train(self):

        """
        nW = open("stopwords.txt", "r")
        neutralWords = [ word for word in nW.read().split() ]
        nW.close()
        """

        totals_W = {}

        # Count files
        numFiles = sum([ len(files) for (r,c,files) in os.walk(_dataSetDir) if os.path.split(r)[-1] != "train" ])
        fileCount = 0

        for (roots, dirs, files) in os.walk(_dataSetDir):

            if len(dirs):
                continue

            fileList = []

            sys.stdout.flush()
            sys.stdout.write("Files read: %d out of %d files\r" % (fileCount, numFiles))

            topic = os.path.split(roots)[-1]

            if topic == "train":
               continue

            self.T_given_D[topic] = 0
            self.W_given_T[topic] = {}

            count = len(files)
            fileCount += count
            self.T_given_D[topic] += count

            for f in files:
                with open("%s/%s" % (roots, f), "r") as fp:

                    # add space.... increases accuracy a little but affects performance

                    file_contents = fp.read()
                    file_contents = re.sub("[\[\]<>\{\}\(\)-.,#:;?!*\"+_=|~`\\/%&0-9]+", " ", file_contents, 0).lower().split()
                    #file_contents = [ word for word in file_contents if word not in neutralWords and len(word) > 2 ]
                    file_contents = [ "%s" % (file_contents[i]) for i in range(len(file_contents) ) ]

                    fileList.append(set(file_conents))

                    for word in file_contents:
                        try:
                           self.W_given_T[topic][word] += 1
                        except:
                           self.W_given_T[topic][word] = 1

            totals_W[topic] = sum( self.W_given_T[topic].values() ) 
        total_T = sum( self.T_given_D.values() )

        # Convert Counts to TF
        for topic in topics:
            self.T_given_D[topic] = self.T_given_D[topic] / float(total_T)

            for word in self.W_given_T[topic].keys():
                self.W_given_T[topic][word] = self.W_given_T[topic][word] / float(totals_W[topic]) #math.log()

        sys.stdout.write( "Files read: %d out of %d files\nTraining Complete\n\n" % (fileCount, numFiles) )


    def calculateProbability(x, mean, stddov):
        exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent


    def writeModelToFile(self, name):

        with open(_modelFile, "w") as fp:

            for key, val in self.T_given_D.iteritems():
                fp.write("%s %s\n" % (key, val))

            for topic in topics:
                fp.write("%s %d\n" % (topic, len(self.W_given_T[topic])))

                for word in self.W_given_T[topic].keys():
                    fp.write("%s %f\n" % (word, self.W_given_T[topic][word]))


    def readModelFile(self, modelFile):

        with open(modelFile, "r") as fp:

            lines = fp.readlines()

            # Read Topic Lines (name, count)
            for row in range(len(topics)):
                line = lines[row].strip().split()
                bayesNet.T_given_D[line[0]] = float(line[1])
            l = len(topics)  
            lines = lines[l:]

            while lines:

                # Read Word Lines
                headerLine = lines[0].strip().split()
                topic, count = headerLine[0], int(headerLine[1])

                bayesNet.W_given_T[topic] = {}

                for row in range(count):
                    line = lines[row].strip().split()
                    bayesNet.W_given_T[topic][line[0]] = float(line[1])

                lines = lines[(count+1):]


    def bayesLaw(self, topic, word):
        top = ( (self.W_given_T[topic][word]) * (self.T_given_D[topic]) ) if word in self.W_given_T[topic] else 0.0
        return top

    def predict(self, contents):

        minValue, predicted = 0.0, None

        for topic in topics:
            #probability = bayesNet.T_given_D[topic] + sum([ (bayesNet.W_given_T[topic][word]) if word in bayesNet.W_given_T[topic] else 0.0 for word in contents ])
            probability = bayesNet.T_given_D[topic]
            for word in contents:
                probability *= (bayesNet.W_given_T[topic][word]) if word in bayesNet.W_given_T[topic] else 1.0 

            if predicted is None or probability > minValue:
                minValue, predicted = probability, topic

        return predicted

""" End of Class """

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.iteritems():
        tfDict[word] = count / float(bowCount)
    return tfDict

def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, count in doc.iteritems():
            if count > 0:
                idfDict[word] += 1

    for word, count in idfDict.iteritems():
        idfDict[word] = math.log(N / float(val))

    return idfDict

def tfidf(tf, idf):
    tfidf = {}
    for word, val in tf.iteritems():
        tfidf[word] = val * idf[word]
    return tfidf


def printAccuracyReport(results_data): 
    print "\n\nConfusion Matrix:"
    print " a\p  " + "".join([ " %5s" % topic for topic in topics ])
    print "      " + "-------"*len(topics)

    for row in range(len(topics)):
        current_row = "%4s |" % topics[row]
        for col in range(len(topics)):
            current_row += " %5s" % int(results_data[topics[row]][topics[col]],)
        print current_row
    print "\n"

    # Accuracy Report
    n = float(sum([ sum(results_data[topic].values()) for topic in topics ]))

    print "Accuracy: ", sum([ results_data[topic][topic] for topic in topics]) / n * 100

    error = 0
    for topic in topics:
        for topic_2 in topics:
            if topic != topic_2:
                error += results_data[topic][topic_2]

    print "Error Rate: ", error / n * 100

    print "             True Rates        False Rates"
    for i in range(len(topics)):

        topic_error = {}
        for topic in topics:
            for topic_2 in topics:
                if topic != topic_2:
                    error += results_data[topic][topic_2]

        print "%8s : %-15s   %-15s" % ( \
              topics[i], \
              results_data[topics[i]][topics[i]] / sum([ results_data[topic_2][topics[i]] for topic_2 in topics]), \
              sum([ results_data[topic_2][topics[i]] for topic_2 in topics if topic_2 != topics[i] ]) / sum([ results_data[topic_2][topics[i]] for topic_2 in topics]))

# =====================================================================================================
# Program Start


_mode = sys.argv[1].lower()
_dataSetDir = sys.argv[2]
_modelFile = sys.argv[3]

topics = ["spam", "ham"]


if not os.path.isdir(_dataSetDir):
    sys.exit("Error! Invalid Data Set Directory.")

bayesNet = BayesNet()

if _mode == "train":
    print "\nTraining...\n"

    bayesNet.train()
    bayesNet.writeModelToFile(_modelFile)

    print "\nWriting distinctive words...\n"

    with open("distinctive_words.txt", "w") as fp:

        for topic in topics:
            bestWords = []
            for word in bayesNet.W_given_T[topic].keys():
                bestWords.append( (word, bayesNet.W_given_T[topic][word]) ) #bayesNet.bayesLaw(topic, word)) )
            bestWords = list(sorted(bestWords, key=lambda tup: tup[1]))
            fp.write("%s\n" % topic)
            for entry in bestWords[:10]:
                fp.write("%s\n" % (entry[0]))
            fp.write("\n")

elif _mode == "test":
    print "\nTesting...\n"

    """
    nW = open("stopwords.txt", "r")
    neutralWords = [ word for word in nW.read().split() ]
    nW.close()
    """

    results_data =  {}

    for topic in topics:
        results_data[topic] = dict.fromkeys(topics, 0.0)

    bayesNet.readModelFile(_modelFile)

    # Count files
    numFiles = sum([ len(files) for (r,d,files) in os.walk(_dataSetDir) ])
    fileCount = 0

    for (roots, dirs, files) in os.walk(_dataSetDir):

        if len(files) < 3:
            continue

        actual = os.path.split(roots)[-1] 

        for f in files:
            fileCount += 1

            sys.stdout.flush()
            sys.stdout.write( "Files read: %d out of %d files\r" % (fileCount, numFiles) )

            with open("%s/%s" % (roots, f), "r") as fp:

                # Iterate through words
                file_contents = fp.read()
                file_contents = re.sub("[\[\]<>\{\}\(\)-.,#:;?!*\"+=_|~`\\/%&0-9]+", " ", file_contents, 0).lower().split()
                #file_contents = [ word for word in file_contents if word not in neutralWords and len(word) > 2 ]
                file_contents = [ "%s" % (file_contents[i]) for i in range(len(file_contents) ) ]

                results_data[actual][bayesNet.predict(file_contents)] += 1

    printAccuracyReport(results_data)

else:
    sys.exit("Error! Invalid mode.")

