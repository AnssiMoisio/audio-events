import utils
import os
import numpy as np
import pickle as pkl
import config


def create_features():
    """
    Create feature vectors from audio files
    """
    for category in config.CATEGORIES:
        path = os.path.join(config.AUDIO_DIR, category)
        os.mkdir(os.path.join(config.MFCC_DIR, "mfccdelta", category))
        file_count = 0
        for audio in os.listdir(path):
            if audio.endswith('.wav'):
                file_count += 1
                y, sr = utils.LoadAudioFile(os.path.join(path, audio))
                mfcc = utils.generateMFCCs(y, sr)
                features = utils.addDeltasAndDeltaDeltas(mfcc)
                with open(os.path.join(config.MFCC_DIR, "mfccdelta", category, str(file_count).zfill(4)+".pkl"),'wb') as f:
                    pkl.dump(features, f)


# create_features()