"""
There are some normalizer classes that are described below.
They server only 2-D data and they deal data in three ways.(overall, rows and columns)
Here is an example of application.
"""

import numpy as np

from Preprocess.Normalizer import MinMaxNormalizer

data = np.array([[1,4,6,8], [3,5,6,2], [8,3,7,10]])
n = MinMaxNormalizer()
data1 = n.normalise(data)  # overall normalization
print(data1)
data2 = n.normalise(data, axis=0)  # rows normalization
print(data2)
data3 = n.normalise(data, axis=1)  # columns normalization
print(data3)
