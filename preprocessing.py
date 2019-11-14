import utils
import glob

for filea in glob.glob("data/*/*.wav"):
    y, sr = utils.LoadAudioFile(filea)
    melspec_dB = utils.generateMelSpecImage(y,sr)
    utils.visulizeFeatures(melspec_dB, sr)
