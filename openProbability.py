
import math
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.linear_model import LogisticRegression


def logit(p):
    return math.log(p / (1 - p))


X_train, y_train = [], []

model = linear_model.LogisticRegression(C=1e5)
model.fit(X_train, y_train)


LogisticRegression(C=100000.0, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, max_iter=100,
                   multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


