#! usr/bin/env/python

import sys

class Bayes_Net():
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def classify(self):
        pass

    def prinConfusionMatrix(self):
        pass


#=======================================================
# Program Start

_mode = sys.argv[1].lower()
_dir = sys.argv[2]
_modelFile = sys.argv[3]

modes = ["train", "test"]

if _mode not in modes:
    sys.exit("Error: invalid mode.")


