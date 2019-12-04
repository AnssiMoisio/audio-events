import numpy as np
import pickle as pkl
import random
import config
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def prepare_data():
    minlen = np.inf
    maxlen = 0
    mfcc_data = []
    for category in range(8):
        mfcc_data.append([])
        path = os.path.join(config.MFCC_DIR, config.CATEGORIES[category])
        for mfcc_pkl in os.listdir(path):
            if mfcc_pkl.endswith(".pkl"):
                with open(os.path.join(path, mfcc_pkl), 'rb') as f:
                    mfcc = pkl.load(f)
                if len(mfcc[0]) < minlen: minlen = len(mfcc[0])
                if len(mfcc[0]) > maxlen: maxlen = len(mfcc[0])
                mfcc_data[category].append(mfcc)

    # pad short audio samples with zeros
    # minlen is the length of the shortest audio/MFCC, maxlen longest
    # print(minlen, maxlen)
    n_bands = len(mfcc_data[category][0])
    padded_mfccs = []
    for category in range(8):
        for mfcc in mfcc_data[category]:
            padded_mfccs.append([np.pad(mfcc, [(0, 0), (0, maxlen - mfcc.shape[1])]), category])

    random.shuffle(padded_mfccs)
    X, Y = [], []
    for i, label in padded_mfccs:
        X.append(i)
        Y.append(label)

    # reshape
    X = np.array(X)
    print(X.shape)
    X = X.reshape((X.shape[0], -1))
    print(X.shape)

    #  scale the data TODO: try normalisation etc
    X = scale(X)

    # split to training, validation and test sets
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.25, train_size =0.75)

    # save
    data = [X_train, Y_train, X_validation, Y_validation, X_test, Y_test]
    iterator = ["X_train", "Y_train", "X_validation", "Y_validation", "X_test", "Y_test"]
    for i, pickle_file in enumerate(iterator):
        with open(os.path.join(config.DATA_DIR, "svm", pickle_file + ".pkl"), "wb") as pklf:
            pkl.dump(data[i], pklf)
