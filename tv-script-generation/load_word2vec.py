
import os
import tensorflow as tf
import numpy as np
from collections import Counter
from itertools import chain

embedding_dim = 100
fname = 'data/glove.6B.%dd.txt'%embedding_dim

glove_index_dict = {}


with open(fname, 'r') as fp:
    glove_symbols = len(fp.readlines())

glove_embedding_weights = np.empty((glove_symbols, embedding_dim))

print("the number of words",glove_symbols)

with open(fname, 'r') as fp:
    i = 0
    for ls in fp:
        ls = ls.strip().split()
        w = ls[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i,:] = np.asarray(ls[1:],dtype=np.float32)
        i += 1

print(glove_embedding_weights[0:2])


