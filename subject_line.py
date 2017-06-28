
import re
import csv
import math
from scipy import stats

"""

Subject -> Open rate
Subject -> CT rate
Content -> CT rate

Features:
    contains certain words
    length of subject (# of words)
    percentage of words capitalized


"""

def t_test(word_1, word_2):

    x1 = word_mean_dict[word_1]
    x2 = word_mean_dict[word_2]
    s1 = stddev(word_1)
    s2 = stddev(word_2)
    n1 = sum([ 1 for row in data if word_1 in row[0] ])
    n2 = sum([ 1 for row in data if word_2 in row[0] ])
    t_val = math.abs(x1 - x2) / math.sqrt( (s1**2) / n1 + (s2**2) / n2 )

    df_arr = [12.71, 4.3, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.2, 2.28, 2.16, 2.14, 2.13, 2.12, 2.11, 2.1, 2.09, 2.09, 2.08, 2.07, 2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.04, 2.04, 2.02, 2.0, 1.98, 1.98]
    df = n1 + n2 - 2
    if df > len(df_arr):
        df = len(df_arr) - 1
    crit_val = df_arr[df]
    return True if t_val > crit_val else False



def stddev(word):
    mean = word_mean_dict[word]
    sublist = [ row[1] for row in data if word in row[0] ]
    return sum([ (score - mean)**2 for score in sublist ]) / len(sublist)


def convertWordToNGram(line, n=2):
    subject = line.split()
    return [ " ".join(subject[i:i+n]) for i in range(len(subject) - n+1) ]


#gram = [ line[j] for j in range(i, n+1+i) ]

f = open("subject_line_data.csv", "r")
reader = csv.reader(f)
data = [ row for row in reader ]
f.close()

avg_open_rate = sum([ float(row[1]) for row in data ]) / len(data)

word_bag = ""
for row in data:
    word_bag += "%s " % row[0]
word_bag = re.sub("[^\w]", " ", word_bag).split()
word_bag = set(word_bag)


word_mean_dict = {}
for word in word_bag:
    total, count = 0, 0
    for i, row in enumerate(data):
        if word in row[0]:
            total += float(row[1])
            count += 1
    if count == 0:
        print word
        continue
    word_mean_dict[word] = total / count

# Get 10 words with highest mean
final = sorted(word_mean_dict.iteritems(), key=lambda (k,v): (v,k))[-10:]

for entry in final:
    print entry

# Find High impact words

for word in final:
    rates_list = [ row[1] for row in data if word[0] in row[0] ]
    t_val = stats.ttest_1samp(rates_list, avg_open_rate)

    crit_val = stats.t.ppf(1-0.05, len(rates_list) - 1)
    if t_val > crit_val:
        print word[0], " high-impact word"


#max(word_mean_dict, key=word_mean_dict.get)

# T test
# 1. collect all words or n-grams used in all the subjects
# 2. perform t test on each and determine which hos consistent high scores
# 3. ...convert into vector and perform multivariate linear regression. Try other forms of regressions


