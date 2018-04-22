"""
Exmaples of how to use ECOC Classifiers.
"""

import numpy as np
from Classifiers.ECOCClassifier import SimpleECOCClassifier
from Classifiers.BaseClassifier import get_base_clf

X = np.array([[.8,.6,.1],[.3,.1,.2],[.5,.9,.3],[0,.5,.8],[.5,.1,.6],[.9,.2,.7]])  # train data
X_ = np.array([[.5,.3,.9],[.7,.1,.4],[.3,.8,0],[.1,0,.9]])  # test data
y = np.array(['A', 'A', 'B', 'B', 'C', 'C'])  # labels
M = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])  # OVO matrix for 3 classes.
estimator = get_base_clf('SVM')  # get SVM classifier object.

sec = SimpleECOCClassifier(estimator, M)
sec.fit(X, y)
pred = sec.predict(X_)

print(pred)
