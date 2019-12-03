import numpy as np
import os
import pickle as pkl
import random
import config

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
# print(minlen, maxlen)
bands = len(mfcc_data[category][0])
padded_mfccs = []
for category in range(8):
    padded_mfccs.append([])
    for mfcc in mfcc_data[category]:
        padded_mfccs[category].append(np.pad(mfcc, [(0, 0), (0, maxlen - mfcc.shape[1])]))



# divide data into training and evaluation sets
train_split = 0.8
for category in range(8):
    random.shuffle(padded_mfccs[category])
    ind = int(len(padded_mfccs[category]) * train_split)
    mfcc_train = padded_mfccs[category][:ind]
    mfcc_eval = padded_mfccs[category][ind:]

    with open (os.path.join(config.DATA_DIR, "train",  config.CATEGORIES[category] + ".pkl"), "wb") as pklf:
        pkl.dump(mfcc_train, pklf)
    with open (os.path.join(config.DATA_DIR, "eval",  config.CATEGORIES[category] + ".pkl"), "wb") as pklf:
        pkl.dump(mfcc_eval, pklf)
        



    
