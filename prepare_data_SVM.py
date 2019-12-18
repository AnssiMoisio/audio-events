import numpy as np
import pickle as pkl
import random
import config
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def prepare_data(features_superfolder):
    minlen = np.inf
    maxlen = 0
    mfcc_data = []
    for category in range(8):
        mfcc_data.append([])
        path = os.path.join(features_superfolder, config.CATEGORIES[category])
        for mfcc_pkl in os.listdir(path):
            if mfcc_pkl.endswith(".pkl"):
                with open(os.path.join(path, mfcc_pkl), 'rb') as f:
                    mfcc = pkl.load(f)
                if len(mfcc[0]) < minlen: minlen = len(mfcc[0])
                if len(mfcc[0]) > maxlen: maxlen = len(mfcc[0])
                mfcc_data[category].append(mfcc)

    # pad short audio samples
    # minlen is the length of the shortest audio/MFCC, maxlen longest
    # print(minlen, maxlen)
    n_bands = len(mfcc_data[category][0])
    padded_mfccs = []
    for category in range(8):
        for mfcc in mfcc_data[category]:
            padded_mfccs.append([np.pad(mfcc, [(0, 0), (0, maxlen - mfcc.shape[1])], 'mean'), category])
    random.shuffle(padded_mfccs)
    X, Y = [], []
    for i, label in padded_mfccs:
        X.append(i)
        Y.append(label)

    # reshape
    X = np.array(X)
    # print(X.shape)
    X = X.reshape((X.shape[0], -1))
    # print(X.shape)

    #  scale the data 
    X = scale(X)

    # save data
    data = [X, Y]
    iterator = ["X", "Y"]
    save_folder = os.path.join(features_superfolder, "_prepared")
    os.mkdir(save_folder)
    for i, pickle_file in enumerate(iterator):
        with open(os.path.join(save_folder, str(pickle_file)+".pkl"), "wb") as pklf:
            pkl.dump(data[i], pklf)

    return save_folder