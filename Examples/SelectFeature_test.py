"""
Feature Selection.
"""

import numpy as np
from FeatureSelection.FeatureSelector import BSSWSS

X = np.array([[.8,.6,.1],[.3,.1,.2],[.5,.9,.3],[0,.5,.8]])  # train data
X_ = np.array([[.5,.3,.9],[.7,.1,.4]])  # test data
y = np.array([1,1,-1,-1])  # labels

fs = BSSWSS(k=2)  # remain 2 features.
fs.fit(X, y)
X, X_ = fs.transform(X), fs.transform(X_)
print('Train data after selection:\n', X)
print('\nTest data after selection:\n', X_)
print('\nRanking indices:\n', fs.get_ranking_indices())
print('\nfeature selected:\n', fs.get_support())
print('\nfeature indices selected:\n', fs.get_support(indices=True))
print('\nfeature scores:\n', fs.scores_)
