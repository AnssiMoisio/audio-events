import numpy as np
import cv2
import os
import pickle as pkl
import random
from sklearn.model_selection import train_test_split
import config

spec_data = []
for category in config.CATEGORIES:
    path = os.path.join(config.SPECTROGRAM_DIR,category)
    class_num = config.CATEGORIES.index(category)
    for spectrogram in  os.listdir(path):
        if spectrogram.endswith(".pkl"):
            with open(path+"/"+spectrogram, 'rb') as f:
                db_spectrogram = pkl.load(f)
            new_array = cv2.resize(db_spectrogram, config.image_size)
            spec_data.append([new_array,class_num])

random.shuffle(spec_data)

X, Y = [], []
for i,label in spec_data:
    X.append(i)
    classArray = [0]*8
    classArray[label] = 1
    Y.append(classArray)

X, X_test, Y, Y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size = 0.25,train_size =0.75)
X_train = np.array(X_train).reshape(-1, config.image_size[0], config.image_size[1], 1)
X_test = np.array(X_test).reshape(-1, config.image_size[0], config.image_size[1], 1)
X_validation = np.array(X_validation).reshape(-1, config.image_size[0], config.image_size[1], 1)
Y_train, Y_validation, Y_test = np.array(Y_train).reshape(-1, 8), np.array(Y_validation).reshape(-1, 8), np.array(Y_test).reshape(-1, 8)
data = [X_train, Y_train, X_validation, Y_validation, X_test, Y_test]
for i,pickle_file in enumerate(["X_train","Y_train","X_validation","Y_validation","X_test","Y_test"]):
    with open("data/"+ pickle_file + ".pkl","wb") as pklf:
        pkl.dump(data[i], pklf)
