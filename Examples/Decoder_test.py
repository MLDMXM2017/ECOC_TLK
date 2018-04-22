"""
There are some decoder examples below.
Decoders calculate the distance matrix. (Samples×Classes)
Distance matrix reveals distances between samples and classes.
Here is an example of application.
"""

import numpy as np
from Decoding.Decoder import AttenuatedEuclideanDecoder, get_decoder

classes = np.array(['A', 'B', 'C'])
M = np.array([[1,1,0],[-1,0,1],[0,-1,-1]])  # OVO matrix for 3 classes.

# Example 1: Attenuated Euclidean Decoder for calculating soft distance.
Y = np.array([[0.6,0.2,-0.8],[-0.4,-0.8,-0.6],[0,0.8,-0.4],[-1,0,0.6]])
decoder = AttenuatedEuclideanDecoder()  # or use code "decoder = get_decoder('AED')".
D = decoder.decode(Y, M)  # distance matrix.
print('Result decoded by AED is: ', classes[D.argmin(axis=1)])

# Example 2: Hamming Decoder only server hard distance.
Y = np.array([[1,0,-1],[-1,-1,-1],[0,1,-1],[-1,0,1]])
decoder = get_decoder('HD')  # or use code "decoder = HammingDecoder()".
print('Result decoded by HD  is: ', classes[decoder.decode(Y, M).argmin(axis=1)])
