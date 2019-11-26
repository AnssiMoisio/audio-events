import os

CATEGORIES = ["Acoustic Guitar", "Bark", "Bell", "Explosion", "Laughter", "Siren", "Sneeze", "Thunder"]
SPECTROGRAM_DIR = "D:/Masters/Speech Recognition/Project/sed-project/data/spectrogram"
AUDIO_DIR   = os.path.join(".", "data", "audio")
MFCC_DIR    = os.path.join(".", "data", "mfccs")

# Parameters
learning_rate = 0.0001
lr_decay = 1e-6
epochs = 30
batch_size = 16
image_size = (256, 128)
n_classes = 8
