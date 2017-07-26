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
    rfi_file = open('/home/hezhaoying/SKA/Data/HTRU_1/save_attrs/RFI.pkl', 'rb')
    pulsar_file = open('/home/hezhaoying/SKA/Data/HTRU_1/save_attrs/pulsars.pkl', 'rb')
    rfi = pickle.load(rfi_file)
    pulsar = pickle.load(pulsar_file)

    print("rfi: ", len(rfi))
    print(type(rfi))
    print()
    print("pulsar: ", len(pulsar))

if __name__ == "__main__":
    read_attrs()
