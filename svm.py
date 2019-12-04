import numpy as np
import pickle as pkl
import config
import os
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import prepare_data_SVM
import preprocessing_HMM_SVM

# preprocessing_HMM_SVM.create_features()
folder = prepare_data_SVM.prepare_data(os.path.join(config.DATA_DIR, "features"))

data = []
for i, pickle_file in enumerate(["X_train","Y_train","X_validation","Y_validation","X_test","Y_test"]):
    with open(os.path.join(folder, pickle_file + ".pkl"), "rb") as pklf:
            data.append(pkl.load(pklf))
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data

svclassifier = svm.SVC(kernel='rbf', gamma='auto', verbose=True, max_iter=-1)
svclassifier.fit(X_train, Y_train)

y_pred = svclassifier.predict(X_validation)

print(confusion_matrix(Y_validation, y_pred))
print(classification_report(Y_validation, y_pred))

