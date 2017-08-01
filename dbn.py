# coding=utf-8

import csv
import numpy as np
import pickle
import os
from datetime import datetime

W = list()
vW = list()  # W为对应的权值，vW用于更新权值

b = list()
vb = list()  # b对应显层的偏置，vb用于更新偏置

c = list()
vc = list()  # c对应隐层的偏置，vc用于更新偏置

hidden_size = [3000, 2000, 1000, 500]
batch_size = 1000
num_epochs = 2
n_ins = 0
momentum = 0
lr = 0.1

record = open('record.txt', 'a')


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


def read_attrs(neg, pos):
    """
    Author: Armstrong He
    Time: 17/7/26
    Read attributes from specified objects and convert to uniform size

    :param neg: negative refers to the path of RFI.pkl
    :param pos: positive refers to the path of pulsars.pkl
    :return: the data used to tran DBN
    """
    rfi_file = open(neg, 'rb')
    pulsar_file = open(pos, 'rb')
    rfi = pickle.load(rfi_file)
    pulsar = pickle.load(pulsar_file)

    rfi_list = list()
    pulsar_list = list()

    for i in range(len(rfi)):
        tmp = rfi[i][:-1]
        label = rfi[i][-1]    # rfi: 0, pulsar: 1
        tmp.extend([0]*(5200 - len(tmp)))
        tmp.append(label)
        rfi_list.append(tmp)

    for i in range(len(pulsar)):
        tmp = pulsar[i][:-1]
        label = pulsar[i][-1]    # rfi: 0, pulsar: 1
        tmp.extend([0]*(5200 - len(tmp)))
        tmp.append(label)
        pulsar_list.append(tmp)

    return rfi_list, pulsar_list


def normalization(data):
    """
    normalize the data
    :param data:
    :return:
    """
    norm_data = list()
    labels = list()
    for i in range(len(data)):
        labels.append(data[i][-1])
        tmp = data[i][:-1]
        max = np.max(tmp)
        min = np.min(tmp)
        for j in range(len(tmp)):
            tmp[j] = (tmp[j] - min)/(max - min)
        norm_data.append(tmp)

    return norm_data, labels


# read data from csv file
def input_data(path, ini=False):
    data = list()
    labels = list()
    c = open(path, 'r')
    for line in csv.reader(c):
        f = str_to_float(line[:-1])
        if f:
            labels.append(int(line[-1]))
            if ini:
                max = np.max(f)
                min = np.min(f)
                for i in range(len(f)):
                    f[i] = (f[i] - min)/(max - min)
            data.append(f)
    return data, labels


def dbn_setup(ins):
    n_ins = ins
    size = [n_ins]
    size.extend(hidden_size)
    n = len(size)  # number of layers in DBN

    for i in range(n-1):
        # W.append(np.zeros([size[i+1], size[i]]))
        # vW.append(np.zeros([size[i+1], size[i]]))
        W.append(np.random.normal(size=[size[i+1], size[i]]))
        vW.append(W[-1])
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
    data = cal_pro(np.add(np.dot(data, np.transpose(W[layer])), np.transpose(c[layer])))
    # data = gibbs(cal_pro(np.add(np.dot(data, np.transpose(W[layer])), np.transpose(c[layer]))))
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
            # print(np.add(np.dot(v1, np.transpose(W[layer])), np.transpose(c[layer])))
            # print("h1_mean: ", h1_mean)
            h1_sample = gibbs(h1_mean)
            # print("h1_sample: ", h1_sample)
            v2_mean = cal_pro(np.add(np.dot(h1_sample, W[layer]), np.transpose(b[layer])))
            # print("v2_mean: ", v2_mean)
            v2_sample = gibbs(v2_mean)
            # print("v2_sample: ", v2_sample)
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


def ext_fea(data, labels, path, catalog):
    num = np.shape(data)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(hidden_size)):
        n = hidden_size[i]
        date_time = str(datetime.now().strftime('%Y/%m/%d_%H:%M:%S'))
        print('%s: Saving %s %dth layer: %d' % (date_time, catalog, i+1, n))
        record.write('%s: Saving %s %dth layer: %d\n' % (date_time, catalog, i+1, n))
        store = data
        r = list()
        tmp = os.path.join(path, '%s_%d_layer_%d.pkl' % (catalog, n, i+1))
        file = open(tmp, 'wb')
        for j in range(num):
            v = store[j]
            h_mean = cal_pro(np.add(np.dot(v, np.transpose(W[i])), np.transpose(c[i])))
            h_sample = gibbs(h_mean)
            result = h_sample.tolist()[0]
            result.append(labels[j])
            r.append(result)
        pickle.dump(r, file, -1)

        # with open("F:\\PycharmProjects\\DBN-python\\Data\\HTRU_1_Combined_Lyon_Thornton_Features(30)_layer_%d.csv" %
        #     (i + 1), "w", newline='') as file:
        #     writer = csv.writer(file)
        #     for j in range(num):
        #         v = store[j]
        #         h_mean = cal_pro(np.add(np.dot(v, np.transpose(W[i])), np.transpose(c[i])))
        #         h_sample = gibbs(h_mean)
        #         data.append(h_sample)
        #         result = h_sample.tolist()[0]
        #         result.append(lables[j])
        #         writer.writerow(result)


def get_train_data(rfi, pulsar):
    rd_rfi = list(range(len(rfi)))
    np.random.shuffle(rd_rfi)
    rd_pulsar = list(range(len(pulsar)))
    np.random.shuffle(rd_pulsar)
    train_data = list()
    for i in range(1000):
        train_data.append(pulsar[rd_pulsar[i]])

    for i in range(1000):
        train_data.append(rfi[rd_rfi[i]])
    print("train_data: ", len(train_data))
    return train_data

if __name__ == "__main__":
    # path = "F:\\PycharmProjects\\DBN-python\\Data\\HTRU_1_Combined_Lyon_Thornton_Features (30)_10_Bin_Discretized.csv"
    # data, labels = input_data(path, True)
    # print("data shape: ", np.shape(data))
    # print("label shape: ", np.shape(labels))
    # x, y = np.shape(data)
    # print(x, y)
    # dbn_setup(ins=y)
    # dbn_train(data[:20000])
    # ext_fea(data, labels)
    # print(W[0][0])
    neg = '/home/ai/SKA/Data/HTRU_1/save_attrs/RFI.pkl'
    pos = '/home/ai/SKA/Data/HTRU_1/save_attrs/pulsars.pkl'
    print("Reading RFI data")
    rfi, pulsars = read_attrs(neg, pos)
    rfi_data, rfi_label = normalization(rfi)
    print("Finishing reading RFI data")
    print('rfi data: ', np.shape(rfi_data))
    print("rfi_label: ", np.shape(rfi_label))
    print('---------------------------------------')
    print('Reading pulsars data')
    pulsars_data, pulsars_label = normalization(pulsars)
    print('Finishing reading pulsar data')
    print("pulsar_data: ", np.shape(pulsars_data))
    print("pulsar_label: ", np.shape(pulsars_label))

    for i in range(1):
        train_data = get_train_data(rfi_data, pulsars_data)
        x, y = np.shape(train_data)
        print("%dth training:%d, %d " % (i, x, y))
        date_time = str(datetime.now().strftime('%Y/%m/%d_%H:%M:%S'))
        record.write("%s: %dth training:%d, %d \n" % (date_time, i, x, y))
        dbn_setup(ins=y)
        dbn_train(data=train_data)
    ext_fea(rfi_data, rfi_label, '/home/hezhaoying/SKA/Data/HTRU_1/RFI_feature_extraction', 'RFI')
    ext_fea(pulsars_data, pulsars_label, '/home/hezhaoying/SKA/Data/HTRU_1/pulsar_feature_extraction', 'pulsar')



