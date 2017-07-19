# coding=utf-8

import csv
import numpy as np
import scipy

W = list()
vW = list()  # W为对应的权值，vW用于更新权值

b = list()
vb = list()  # b对应显层的偏置，vb用于更新偏置

c = list()
vc = list()  # c对应隐层的偏置，vc用于更新偏置

hidden_size = [100, 100]
batch_size = 1000
num_epochs = 2
n_ins = 0
momentum = 0
lr = 0.1


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def cal_pro(x):
    m, n = np.shape(x)
    for i in range(m):
        for j in range(n):
            x[i][j] = sigmoid(x[i][j])
    return x


def gibbs(x):
    pro = np.random.random()
    m, n = np.shape(x)
    for i in range(m):
        for j in range(n):
            if x[i][j] >= pro:
                x[i][j] = 1
            else:
                x[i][j] = 0
    return x


# convert string to float
def str_to_float(raw):
    f = []
    for i in range(len(raw)):
        if raw[i]:
            f.append(float(raw[i]))
        else:
            return
    return f


# read data from csv file
def input_data(path, ini=False):
    data = list()
    c = open(path, 'r')
    for line in csv.reader(c):
        f = str_to_float(line[:-1])
        if f:
            if ini:
                max = np.max(f)
                min = np.min(f)
                for i in range(len(f)):
                    f[i] = (f[i] - min)/(max - min)
            data.append(f)
    return data


def dbn_setup(ins):
    n_ins = ins
    size = [n_ins]
    size.extend(hidden_size)
    n = len(size)  # number of layers in DBN

    for i in range(n-1):
        W.append(np.zeros([size[i+1], size[i]]))
        vW.append(np.zeros([size[i+1], size[i]]))
        b.append(np.zeros([size[i], 1]))
        vb.append(np.zeros([size[i], 1]))
        c.append(np.zeros([size[i+1], 1]))
        vc.append(np.zeros([size[i+1], 1]))


def dbn_train(data):
    n = len(hidden_size)

    # 训练第一个RBM
    rbm_train(data, layer=0)

    for i in range(1, n):
        # 建立RBM
        data = rbm_up(data, i-1)
        # 训练RBM
        # print(np.shape(data))
        rbm_train(data, i)


# 上一层的RBM的输出作为下一层RBM的输入
def rbm_up(data, layer):
    # data = cal_pro(np.add(np.dot(data, np.transpose(W[layer])), np.transpose(c[layer])))
    data = gibbs(cal_pro(np.add(np.dot(data, np.transpose(W[layer])), np.transpose(c[layer]))))
    return data


# layer 代表训练第layer+1个RBM
def rbm_train(data, layer):
    num = np.shape(data)[0]    # number of training set
    num_batchs = num//batch_size
    err = 0
    print("Starting training ", layer+1, "th RBM")
    print("err: ", err)
    for i in range(num_epochs):
        # 产生num-1的随机排序
        rd_id = list(range(num))
        np.random.shuffle(rd_id)

        for j in range(num_batchs):
            batch = list()
            for k in range(batch_size):
                batch.append(data[rd_id[j*batch_size + k]])


            # CD-k, k=1
            v1 = batch
            # print("c: ", np.shape(c[layer]), np.shape(W[layer]), np.shape(v1))
            h1_mean = cal_pro(np.add(np.dot(v1, np.transpose(W[layer])), np.transpose(c[layer])))

            h1_sample = gibbs(h1_mean)
            v2_mean = cal_pro(np.add(np.dot(h1_sample, W[layer]), np.transpose(b[layer])))
            v2_sample = gibbs(v2_mean)
            h2_mean = cal_pro(np.add(np.dot(v2_sample, np.transpose(W[layer])), np.transpose(c[layer])))
            h2_sample = gibbs(h2_mean)

            c1 = np.dot(np.transpose(h1_sample), v1)
            c2 = np.dot(np.transpose(h2_mean), v2_sample)
            # c2 = np.dot(np.transpose(h2_sample), v2_sample)

            # print('before V:', np.shape(vW[layer]), np.shape(vb[layer]), np.shape(vc[layer]))
            # print(np.shape(np.transpose(np.divide(np.dot(np.transpose(np.sum(np.subtract(v1, v2_sample), axis=0)), lr), batch_size))))
            vW[layer] = np.add(np.dot(momentum, vW[layer]), np.divide(np.dot(np.subtract(c1, c2), lr), batch_size))
            vb[layer] = np.add(np.dot(momentum, vb[layer]), np.transpose(np.divide(np.dot(np.transpose(np.sum(np.subtract(v1, v2_sample), axis=0)), lr), batch_size)).reshape(np.shape(vb[layer])))
            vc[layer] = np.add(np.dot(momentum, vc[layer]), np.transpose(np.divide(np.dot(np.transpose(np.sum(np.subtract(h1_sample, h2_mean), axis=0)), lr), batch_size)).reshape(np.shape(vc[layer])))
            # vc[layer] = np.add(np.dot(momentum, vc[layer]), np.transpose(
            #     np.divide(np.dot(np.transpose(np.sum(np.subtract(h1_sample, h2_sample), axis=0)), lr),
            #               batch_size)).reshape(np.shape(vc[layer])))

            # print('after V:', np.shape(vW[layer]), np.shape(vb[layer]), np.shape(vc[layer]))
            W[layer] = W[layer] + vW[layer]
            b[layer] = b[layer] + vb[layer]
            c[layer] = c[layer] + vc[layer]
            # print(np.shape(c[layer]))
            err = err + np.sum(np.subtract(v1, v2_sample)*np.subtract(v1, v2_sample))/batch_size
            # err = err + np.sum(np.subtract(v1, v2_mean) * np.subtract(v1, v2_mean)) / batch_size
            print("epoch: ", i+1, " batch: ", j+1, " err: ", err)


if __name__ == "__main__":
    path = "D:\\Arm\\PyCharmProject\\Data\\HTRU_1_Combined_Lyon_Thornton_Features (30).csv"
    data = input_data(path, True)
    x, y = np.shape(data)
    dbn_setup(ins=y)
    dbn_train(data[:])
    print(W[0][0])