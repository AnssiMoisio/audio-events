import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

def LoadAudioFile(fileName):
    return librosa.load(fileName)

"""
With the librosa default sampling rate of 22050Hz, a window of 1024 samples
means about 46ms. This is similar with Cakir et al. 2017: "Convolutional
Recurrent Neural Networks for Polyphonic Sound Event Detection"

If unspecified, window defaults to win_length = n_fft
"""
def generateMelSpecImage(y, sr, n_fft=1024, hop_length=512, n_mels=40):
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels), ref=np.max)

# the same as above but just more explicit for transparency
def generateMelSpecImage_explicit(y, sr, n_fft=1024, hop_length=512, n_mels=40):
    spec = np.power(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), 2)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return librosa.power_to_db(np.dot(mel_basis, spec), ref=np.max)

def generateLogMelEnergies(y, sr, n_fft=1024, hop_length=512, n_mels=40):
    """
    logarithmic mel band energies as in Cakir et al. (2017)

    TODO: Cakir et al.: "each energy band is normalized by subtracting its mean and
    dividing by its standard deviation computed over the training
    set. The normalized log mel band energies are finally split
    into sequences"
    """
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return np.log(np.dot(mel_basis, spec))

def generateMFCCs(y, sr, n_mfcc=40):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def visulizeFeatures(image, sr, title='Mel-frequency spectrogram'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(image, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
