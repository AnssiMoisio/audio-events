import numpy as np
import librosa
from pomegranate import *
import pickle as pkl
import config

## mixture model components
# d1 = MultivariateGaussianDistribution([1, 6, 3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# d2 = MultivariateGaussianDistribution([2, 8, 4], [[1, 0, 0], [0, 1, 0], [0, 0, 2]])
# d3 = MultivariateGaussianDistribution([0, 4, 8], [[2, 0, 0], [0, 3, 0], [0, 0, 1]])
## create mixture model
# mixture_model = GeneralMixtureModel([d1, d2, d3], weights=[0.25, 0.60, 0.15])

mean = np.array([0.0] * 40)
covariance = np.eye(40)
d = MultivariateGaussianDistribution(mean, covariance)
mixture_model = GeneralMixtureModel([d, d, d], weights=[0.35, 0.35, 0.30])

# 9 states as in Ma et al. (2006)
states = []
for i in range(9):
    states.append(State(mixture_model))

# create HMM
model = HiddenMarkovModel()
model.add_states(states)
model.add_transition(model.start, states[0], 1.0)
for i in range(len(states) - 1):
    model.add_transition(states[i], states[i], 0.4)
    model.add_transition(states[i], states[i+1], 0.6)
model.add_transition(states[i+1], states[i+1], 0.4)
model.add_transition(states[i+1], model.end, 0.6)
model.bake()

# train
mfcc_data = []
path = os.path.join(config.MFCC_DIR, config.CATEGORIES[0])
for mfcc_pkl in os.listdir(path):
    if mfcc_pkl.endswith(".pkl"):
        with open(os.path.join(path, mfcc_pkl), 'rb') as f:
            mfcc = pkl.load(f)
        mfcc_data.append(mfcc)

print(len(mfcc_data), "mfccs / audio samples")
print(len(mfcc_data[0]), "frequency bands")
print(len(mfcc_data[0][0]), "time steps in the first mfcc")

model.fit( mfcc_data, max_iterations=500 )

print( model.log_probability( mfcc_data[3] ) )