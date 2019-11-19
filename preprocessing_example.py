import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import utils

y, sr = librosa.load('sound_example.wav')
print("librosa default sampling rate:", sr)
melspec_dB = utils.generateMelSpecImage(y, sr)
mfccs = utils.generateMFCCs(y, sr)
mbe = utils.generateLogMelEnergies(y, sr)

melspec_dB2 = utils.generateMelSpecImage_explicit(y, sr)
if melspec_dB2.all() == melspec_dB.all(): print("melspec_dB1 is melspec_dB2")

# plot mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec_dB, x_axis='time',y_axis='mel', sr=sr, fmax=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()

# plot MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.tight_layout()

# plot log mel band energies
plt.figure(figsize=(10, 4))
librosa.display.specshow(mbe, x_axis='time',y_axis='mel', sr=sr, fmax=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('logarithmic mel band energies')
plt.tight_layout()

plt.show()