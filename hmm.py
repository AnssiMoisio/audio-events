import numpy as np
import librosa
from pomegranate import *
from pomegranate import utils as pom_utils
import pickle as pkl
import config
from sklearn.cluster import KMeans
import time

# pom_utils.enable_gpu()

SAVED_MODELS_DIR = os.path.join(".", "saved models")

# load data
train_data = []
eval_data = []
for category in range(8):
    with open(os.path.join(config.DATA_DIR, "train", config.CATEGORIES[category] + ".pkl"), "rb") as pklf:
        train_data.append(pkl.load(pklf))
    with open(os.path.join(config.DATA_DIR, "eval", config.CATEGORIES[category] + ".pkl"), "rb") as pklf:
        eval_data.append(pkl.load(pklf))

# move these to config.py
n_states_in_hmm = 3
n_components_in_mixture = 3


# mfcc for initialisation
X = np.array(train_data[0][0])
# print(X.shape) # (40, 431)
X = np.transpose(X)
# print(X.shape) # (431, 40)

X = np.array(train_data[0])
print(X.shape)
initx = np.reshape(X, (X.shape[0]*X.shape[2], X.shape[1]))
print(initx.shape)


print("Initialising the model:")
print("K-means clustering of states...")
# initialise the 10 states for category "acoustic guitar"
clusters = KMeans(n_states_in_hmm).fit_predict(initx)

print("K-means clustering of states complete")
print("Clustering of gaussian mixtures...")
# initialise the gaussian distributions for each state
distributions = []
for i in range(n_states_in_hmm):
    print("Creating distribution for HMM state", i)
    X_subset = initx[clusters == i]
    print("This state cluster size in the initialisation:", len(X_subset))
    distribution = GeneralMixtureModel.from_samples(NormalDistribution, n_components=n_components_in_mixture, X=X_subset)
    distributions.append(distribution)
print("Clustering of gaussian mixtures complete")

filename = "hmm_init" + time.strftime("%Y%m%d-%H%M%S") + ".pkl"
with open(os.path.join(SAVED_MODELS_DIR, filename), 'wb') as f:
    pkl.dump(distributions, f) 

transitions = np.ones((n_states_in_hmm, n_states_in_hmm), dtype='float64') / n_states_in_hmm
starts = np.ones(n_states_in_hmm, dtype='float64') / n_states_in_hmm
model = HiddenMarkovModel.from_matrix(transitions, distributions, starts, verbose=True)

print("Initialisation complete")


# train
n_samples = len(train_data[0])
X_train = np.array(train_data[0]) #.reshape((n_samples, -1))
print(X_train.shape)
# X_train = np.swapaxes(X_train,1,2)
# print(X_train.shape)

print("Training the model...")
model.fit(X_train, algorithm="baum-welch", verbose=True)#, max_iterations=10)
print("Training complete")

filename = "hmm_model" + time.strftime("%Y%m%d-%H%M%S") + ".pkl"
with open(os.path.join(SAVED_MODELS_DIR, filename), 'wb') as f:
    pkl.dump(model, f) 

'''

# evaluate
with open(os.path.join(SAVED_MODELS_DIR, "hmm_model20191203-192334.pkl"), "rb") as pklf:
    model = pkl.load(pklf)

print(model.log_probability(eval_data[0][0]))


for category in range(8):
    probs = np.zeros((len(eval_data[category])))
    for i in range(len(probs)):
        probs[i] = model.log_probability(eval_data[category][i])
    print(config.CATEGORIES[category], np.mean(probs))
'''