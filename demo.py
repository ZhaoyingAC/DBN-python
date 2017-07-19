# coding=utf-8

import tensorflow as tf
import numpy as np


l = [[1, 2, 3], [5, 6, 7], [8, 9, 0], [3, 2, 7]]
n = list(range(3))
x, y = np.shape(l)
m = np.ones([3, ])
print(x, y)
print(np.shape(l))
np.random.shuffle(n)
print(n)
batch = list()
for i in range(3):
    batch.append(l[n[i]])
print(batch)
print(np.add(l, m))
a = [[1, 2, 3], [4, 5, 6]]
c = [[1, 1, 1], [2, 2, 2]]
b = [[1, 2], [2, 3], [3, 4]]
print(np.dot(2, b))
print(np.subtract(a, c))
print(np.add(a, c))
print(np.sum(np.subtract(a, c)*np.subtract(a, c))/5)
print(np.divide(c, 2))
print(np.add(b, m))
