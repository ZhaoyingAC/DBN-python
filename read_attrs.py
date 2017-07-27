# coding=utf-8

"""
Author: Armstrong He
Time: 17/7/26

Read attributes from specified objects and convert to uniform size
"""

import pickle


def read_attrs(neg, pos):
    """
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

    for i in range(rfi):
        tmp = rfi[i][:-1]
        label = rfi[i][-1]    # rfi: 0, pulsar: 1
        tmp.extend([0]*(6000 - len(tmp)))
        tmp.appen(label)
        rfi_list.append(tmp)

    for i in range(pulsar):
        tmp = pulsar[i][:-1]
        label = pulsar[i][-1]    # rfi: 0, pulsar: 1
        tmp.extend([0]*(6000 - len(tmp)))
        tmp.appen(label)
        pulsar_list.append(tmp)

    return rfi_list, pulsar_list



if __name__ == "__main__":
    neg = '/home/hezhaoying/SKA/Data/HTRU_1/save_attrs/RFI.pkl'
    pos = '/home/hezhaoying/SKA/Data/HTRU_1/save_attrs/pulsars.pkl'
    read_attrs(neg, pos)
