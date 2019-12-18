import os

CATEGORIES = ["Acoustic Guitar", "Bark", "Bell", "Explosion", "Laughter", "Siren", "Sneeze", "Thunder"]
SPECTROGRAM_DIR = "D:/Masters/Speech Recognition/Project/sed-project/data/spectrogram"

DATA_DIR    = os.path.join(".", "data", "data-full")
AUDIO_DIR   = os.path.join(DATA_DIR, "audio")

# Parameters
learning_rate = 0.0001
lr_decay = 1e-6
epochs = 30
batch_size = 16
image_size = (256, 128)
n_classes = 8


# Pre-processing parameters
"""
With the librosa default sampling rate of 22050Hz, a window of 1024 samples
means about 46ms. This is similar with Cakir et al. 2017: "Convolutional
Recurrent Neural Networks for Polyphonic Sound Event Detection"

If unspecified, window defaults to win_length = n_fft
"""
n_fft = 1024
hop_length = 512
n_mels = 40 
