"""
There are some examples of how to use Data Complexities.
Data Complexities are measures of separability of a two-class data.
SMALL SCORES MEAN LOW DATA COMPLEXITIES AND HIGH SEPARABILITY.
Here is an example of application.
"""

import numpy as np
from DataComplexity.datacomplexity import get_data_complexity

X = np.array([[.8,.6,.1],[.3,.1,.2],[.5,.9,.3],[0,.5,.8]])
y = np.array([1,1,-1,-1])  # Two-class labels of data.

dc = get_data_complexity('F1')
print('Score of F1 data complexity: %.4f' % dc.score(X, y))
dc = get_data_complexity('F2')
print('Score of F2 data complexity: %.4f' % dc.score(X, y))
dc = get_data_complexity('F3')
print('Score of F3 data complexity: %.4f' % dc.score(X, y))
