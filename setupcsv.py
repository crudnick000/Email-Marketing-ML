
"""
    Tips:
    most people think subject line is the most important thing, it's really the second. Who is first.

    1. main benefit
        parse subject, remove "How to", "# Ways to", "# Steps to", ect.

        Does All Caps work? All words are capitalized?

        Does it match the content

        Can we measure trust

    2. "You are not alone..."
        90% open rate?

    3. "Hey..."
        30% open rate

    4. "Watch this video"
        people like to watch video

    5. "You'll love this..."

    6. "Have you seen this?"


    in General or within a specific industry

    ===============

    for each campaign, calculate the open rate and standardize is with open rate and stddev

    remove special symbols and lowercase all subject lines

    for any given word

"""

import os
import csv
import random

path = "./data/train/Enron/enron1/ham/1691.2000-07-20.farmer.ham.txt"

subjects = []
open_rates = []

for (roots, dirs, files) in os.walk("./data/train/"):
    if len(files) < 3:
        continue

    for f in files:
        fp = open(roots + "/" + f, "r")
        line = fp.readline().lower().split()[1:]
        subjects.append(" ".join(line))
        fp.close()

        open_rates.append(round(random.uniform(0, 1), 4))

fpw = open("./subjectline_data.csv", "w")
writer = csv.writer(fpw)

for i in range(len(subjects)):
    writer.writerow([subjects[i], open_rates[i]])

fpw.close()






