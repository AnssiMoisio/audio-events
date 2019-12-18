import utils
import os
import numpy as np
import pickle as pkl
import config
from shutil import copyfile, copy2 


def create_features(audio_dir, features_dir):
    """
    Create feature vectors from audio files
    """
    print("Create features: \n", "melbands: ", config.n_mels, ", win length: ", config.n_fft)
    superfolder = os.path.join(features_dir, "melbands"+str(config.n_mels)+"winlength"+str(config.n_fft))
    os.mkdir(superfolder)
    for category in config.CATEGORIES:
        audio_path = os.path.join(audio_dir, category)
        features_path = os.path.join(superfolder, category)
        os.mkdir(features_path)
        file_count = 0
        for audio in os.listdir(audio_path):
            if audio.endswith('.wav'):
                file_count += 1
                y, sr = utils.LoadAudioFile(os.path.join(audio_path, audio))

                #### create features here ####
                features = utils.generateMFCCs(y, sr)
                # deltas = utils.addDeltasAndDeltaDeltas(mfccs)
                # zcr = utils.zcr(y)
                # rms = utils.rms(y)
                # features = np.vstack((mfccs, rms))
                # features = utils.generateMelSpecImage(y, sr)
                with open(os.path.join(features_path, str(file_count).zfill(4)+".pkl"),'wb') as f:
                    pkl.dump(features, f)

    return superfolder


def divide_data(train_split=0.8):
    """
    Divide data to train and test sets.
    """
    training_folder = os.path.join(config.DATA_DIR, "training", "audio-small")
    eval_folder = os.path.join(config.DATA_DIR, "eval", "audio-small")
    
    for category in config.CATEGORIES:
        audio_path = os.path.join(config.AUDIO_DIR, category)
        category_path_train = os.path.join(training_folder, category)
        category_path_eval = os.path.join(eval_folder, category)
        os.mkdir(category_path_train)
        os.mkdir(category_path_eval)
        n_train = int(len(os.listdir(audio_path)) * train_split)
        for file_count, audio in enumerate(os.listdir(audio_path)):
            if audio.endswith('.wav'):
                # if file_count < n_train:
                if file_count < 140:
                    copy2(os.path.join(audio_path, audio), category_path_train)
                # else:
                elif file_count > 139 and file_count < 175:
                    copy2(os.path.join(audio_path, audio), category_path_eval)
                    
