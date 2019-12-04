import utils
import os
import numpy as np
import pickle as pkl
import config


def create_MFCCs():
    """
    Create MFCCs from audio files
    """
    for category in config.CATEGORIES:
        path = os.path.join(config.AUDIO_DIR, category)
        file_count = 0
        for audio in os.listdir(path):
            if audio.endswith('.wav'):
                file_count += 1
                y, sr = utils.LoadAudioFile(os.path.join(path, audio))
                mfcc = utils.generateMFCCs(y, sr)
                with open(os.path.join(config.MFCC_DIR, category, str(file_count).zfill(4)+".pkl"),'wb') as f:
                    pkl.dump(mfcc, f)


# TODO: add deltas and delta-deltas (and possibly other features) to the feature vectors
def create_deltas():
    pass