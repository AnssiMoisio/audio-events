import utils
import glob
import os
import numpy as np
import pickle as pkl
import cv2

AUDIO_DIR = "D:/Masters/Speech Recognition/Project/sed-project/data/audio"
CATEGORIES = ["Acoustic Guitar", "Bark", "Bell", "Explosion", "Laughter", "Siren", "Sneeze", "Thunder"]
SPECTROGRAM_DIR = "D:/Masters/Speech Recognition/Project/sed-project/data/spectrogram_40"
a = []
for category in CATEGORIES:
    path = os.path.join(AUDIO_DIR,category)
    category_count = 0
    for audio in  os.listdir(path):
        if audio.endswith('.wav'):
            #print(audio,str(category_count).zfill(4))
            category_count+=1
            y, sr = utils.LoadAudioFile(path+"/"+audio)
            db_spectrogram = utils.generateMelSpecImage(y,sr,n_mels=40)

            #print(np.mean(db_spectrogram),np.min(db_spectrogram),np.max(db_spectrogram),type(db_spectrogram))
            #print(np.mean(new_db_spectrogram),np.min(new_db_spectrogram),np.max(new_db_spectrogram),type(new_db_spectrogram))
            with open(os.path.join(SPECTROGRAM_DIR,category,str(category_count).zfill(4)+".pkl"),'wb') as f:
                pkl.dump(db_spectrogram, f)
            print(np.mean(db_spectrogram),np.min(db_spectrogram),np.max(db_spectrogram),db_spectrogram.shape)
