import numpy as np
import pickle as pkl
import config
import os
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import prepare_data_SVM
import preprocessing_HMM_SVM
import confusion # ominous

# create data
dir_train = preprocessing_HMM_SVM.create_features(os.path.join(config.DATA_DIR, "training", "audio-small"), os.path.join(config.DATA_DIR, "training", "features-small"))
dir_eval = preprocessing_HMM_SVM.create_features(os.path.join(config.DATA_DIR, "eval", "audio-small"), os.path.join(config.DATA_DIR, "eval", "features-small"))

# use existing data
# features_folder = os.path.join(config.FEATURES_DIR, "mfccs", "win2048melbands40")

# prepare data
prepare_data_SVM.prepare_data(dir_train)
prepare_data_SVM.prepare_data(dir_eval)
train_folder = os.path.join(os.path.join(dir_train, "_prepared"))
eval_folder = os.path.join(os.path.join(dir_eval, "_prepared"))

# use existing data
# train_folder = os.path.join(config.DATA_DIR, "training", "features", "melbands40winlength10240", "_prepared")
# eval_folder = os.path.join(config.DATA_DIR, "eval", "features", "melbands40winlength10240", "_prepared")

train_data = []
for i, pickle_file in enumerate(["X", "Y"]):
    with open(os.path.join(train_folder, pickle_file + ".pkl"), "rb") as pklf:
            train_data.append(pkl.load(pklf))
X_train, Y_train = train_data
eval_data = []
for i, pickle_file in enumerate(["X", "Y"]):
    with open(os.path.join(eval_folder, pickle_file + ".pkl"), "rb") as pklf:
            eval_data.append(pkl.load(pklf))
X_eval, Y_eval = eval_data

# train
print("Training SVM, data shape:", X_train.shape)
svclassifier = svm.SVC(kernel='rbf', gamma='auto', verbose=False, max_iter=-1)
svclassifier.fit(X_train, Y_train)

# evaluate
y_pred = svclassifier.predict(X_eval)
cm = confusion_matrix(Y_eval, y_pred)
print(classification_report(Y_eval, y_pred))
print(cm)
