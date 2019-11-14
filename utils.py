import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

def LoadAudioFile(fileName):
    return librosa.load(fileName)

def generateMelSpecImage(y, sr):
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)

def visulizeFeatures(image, sr, title='Mel-frequency spectrogram'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(image, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
