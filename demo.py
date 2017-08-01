# coding=utf-8

import csv
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
d = [1, 2, 3]
e = np.array(b)
print(np.dot(2, b))
print(np.subtract(a, c))
print(np.add(a, c))
print(np.sum(np.subtract(a, c)*np.subtract(a, c))/5)
print(np.divide(c, 2))

print(np.add(b, np.array(d).reshape([3, 1])))
store = list()
print(store)
store = b
print(store)
print(b[1:np.shape(b)[0]])
# path = "F:\\PycharmProjects\\DBN-python\\Data\\HTRU_1_Combined_Lyon_Thornton_Features (30)_10_Bin_Discretized.csv"
# for i in range(3):
#     with open("HTUR_1_layer_%d.csv"%(i), "w", newline='') as file:
#         writer = csv.writer(file)
#         for j in range(len(e)):
#             print(e[j])
#             print(type(e[j]))
#             writer.writerow(e[j])

# import pickle
# data = [1, 2, 3, 4, 5, 6]
# data1 = {'a': 1, 'b': 2}
#
# file = open("data.pkl", "rb")
#
# # pickle.dump(data, file, -1)
# # pickle.dump(data1, file, -1)
#
# # data = pickle.load(file)
# for data in pickle.load(file):
#     print("data: ", data)

from sklearn import svm
print('----------------------------------------')
X = [[0, 0], [1, 1], [1, 0]]  # training samples
y = [0, 1, 1]  # training target
clf = svm.SVC()  # class
clf.fit(X, y)  # training the svc model

result = clf.predict([[2, 2], [0,  0], [1, 3]])  # predict the target of testing samples
print(result)  # target

print(clf.support_vectors_)  # support vectors

print(clf.support_)  # indeices of support vectors

print(clf.n_support_)  # number of support vectors for each class

print('-----------------------------------')
X = [[0, 0], [1, 1]]
y = [0.5, 1.5]
clf = svm.SVR()
clf.fit(X, y)
result = clf.predict([[2, 2]])
print(result)