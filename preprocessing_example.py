import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('sound_example.wav', duration=2.0)
# y, sr = librosa.load(librosa.util.example_audio_file())
melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
melspec_dB = librosa.power_to_db(melspec, ref=np.max)

# plot mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec_dB, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()

# plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.tight_layout()
plt.show()