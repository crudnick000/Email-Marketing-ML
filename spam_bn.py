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
    def __init__(self, dataDir):
        self.dataDir = dataDir
        self.T_given_D = {}
        self.W_given_T = {}
        self.classes = ["spam", "ham"]


    def train(self):

        """
        nW = open("stopwords.txt", "r")
        neutralWords = [ word for word in nW.read().split() ]
        nW.close()
        """

        totals_W = {}

        # Count files
        numFiles = sum([ len(files) for (r,c,files) in os.walk(self.dataDir) if os.path.split(r)[-1] != "train" ])
        fileCount = 0

        for (roots, dirs, files) in os.walk(self.dataDir):

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

                    fileList.append(set(file_contents))

                    for word in file_contents:
                        try:
                           self.W_given_T[topic][word] += 1
                        except:
                           self.W_given_T[topic][word] = 1

            totals_W[topic] = sum( self.W_given_T[topic].values() ) 
        total_T = sum( self.T_given_D.values() )

        # Convert Counts to TF
        for topic in self.classes:
            self.T_given_D[topic] = self.T_given_D[topic] / float(total_T)

            for word in self.W_given_T[topic].keys():
                self.W_given_T[topic][word] = self.W_given_T[topic][word] / float(totals_W[topic]) #math.log()

        sys.stdout.write( "Files read: %d out of %d files\nTraining Complete\n\n" % (fileCount, numFiles) )


    def calculateProbability(x, mean, stddov):
        exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent


    def writeModelToFile(self, name):

        with open(name, "w") as fp:

            for key, val in self.T_given_D.iteritems():
                fp.write("%s %s\n" % (key, val))

            for topic in self.classes:
                fp.write("%s %d\n" % (topic, len(self.W_given_T[topic])))

                for word in self.W_given_T[topic].keys():
                    fp.write("%s %f\n" % (word, self.W_given_T[topic][word]))


    def readModelFile(self, modelFile):

        with open(modelFile, "r") as fp:

            lines = fp.readlines()

            # Read Topic Lines (name, count)
            for row in range(len(self.classes)):
                line = lines[row].strip().split()
                self.T_given_D[line[0]] = float(line[1])
            l = len(self.classes)
            lines = lines[l:]

            while lines:

                # Read Word Lines
                headerLine = lines[0].strip().split()
                topic, count = headerLine[0], int(headerLine[1])

                self.W_given_T[topic] = {}

                for row in range(count):
                    line = lines[row].strip().split()
                    self.W_given_T[topic][line[0]] = float(line[1])

                lines = lines[(count+1):]


    def bayesLaw(self, topic, word):
        top = ( (self.W_given_T[topic][word]) * (self.T_given_D[topic]) ) if word in self.W_given_T[topic] else 0.0
        return top

    def predict(self, contents):

        minValue, predicted = 0.0, None

        for topic in self.classes:
            #probability = bayesNet.T_given_D[topic] + sum([ (bayesNet.W_given_T[topic][word]) if word in bayesNet.W_given_T[topic] else 0.0 for word in contents ])
            probability = self.T_given_D[topic]
            for word in contents:
                probability *= (self.W_given_T[topic][word]) if word in self.W_given_T[topic] else 1.0 

            if predicted is None or probability > minValue:
                minValue, predicted = probability, topic

        return predicted

    def printAccuracyReport(self, results_data):

        print "\n\nConfusion Matrix:"
        print " a\p  " + "".join([ " %5s" % topic for topic in self.classes ])
        print "      " + "-------"*len(self.classes)

        for row in range(len(self.classes)):
            current_row = "%4s |" % self.classes[row]
            for col in range(len(self.classes)):
                current_row += " %5s" % int(results_data[self.classes[row]][self.classes[col]],)
            print current_row
        print "\n"

        # Accuracy Report
        n = float(sum([ sum(results_data[topic].values()) for topic in self.classes ]))

        print "Accuracy: ", sum([ results_data[topic][topic] for topic in self.classes ]) / n * 100

        error = 0
        for topic in self.classes:
            for topic_2 in self.classes:
                if topic != topic_2:
                    error += results_data[topic][topic_2]

        print "Error Rate: ", error / n * 100, "\n"
        print "             True Rates        False Rates"
        for i in range(len(self.classes)):

            topic_error = {}
            for topic in self.classes:
                for topic_2 in self.classes:
                    if topic != topic_2:
                        error += results_data[topic][topic_2]

            print "%8s : %-15s   %-15s" % ( \
                  self.classes[i], \
                  results_data[self.classes[i]][self.classes[i]] / sum([ results_data[topic_2][self.classes[i]] for topic_2 in self.classes]), \
                  sum([ results_data[topic_2][self.classes[i]] for topic_2 in self.classes if topic_2 != self.classes[i] ]) / sum([ results_data[topic_2][self.classes[i]] for topic_2 in self.classes]))
        print "\n"

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


# =====================================================================================================

def main(mode, dataDir, modelFile):

    _mode = mode
    _dataSetDir = dataDir
    _modelFile = modelFile

    topics = ["spam", "ham"]

    if not os.path.isdir(_dataSetDir):
        sys.exit("Error! Invalid Data Set Directory.")

    bayesNet = BayesNet(_dataSetDir)

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

        bayesNet.printAccuracyReport(results_data)

    else:
        sys.exit("Error! Invalid mode.")


if __name__ == "__main__":
    _mode = sys.argv[2].lower()
    _dataSetDir = sys.argv[3].lower()
    _modelFile = sys.argv[4].lower()
    main(_mode, _dataSetDir, _modelFile)


