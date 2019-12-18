import csv
import os
import config

"""
Remove audio samples that have more than one label from the 8 classes.
"""

classes = { "/m/05tny_": "Bark",
            "/m/01hsr_": "Sneeze",
            "/m/0395lw": "Bell",
            "/m/042v_gx": "Acoustic Guitar",
            "/m/0ngt1": "Thunder",
            "/m/014zdl": "Explosion",
            "/m/03kmc9": "Siren",
            "/m/01j3sz": "Laughter"}

in_list = list(csv.reader(open("unbalanced_train_segments.csv","r")))
multiclasses = set()
a = 0
for line in in_list:
    try:
        labels = line[3:]
        labels[0] = labels[0][2:]
        labels[-1] = labels[-1][:-1]
        labels = set(labels)
        if len(labels.intersection(classes.keys())) > 1:
            multiclasses.add(line[0])
            a += 1
            # print(line[0], labels.intersection(classes.keys()))
    except IndexError:
        pass

print(a, len(multiclasses))

audio_dir = os.path.join(config.DATA_DIR, "audio")

for category in config.CATEGORIES:
    audio_path = os.path.join(audio_dir, category)
    for audio in os.listdir(audio_path):
        if audio[:11] in multiclasses:
            print("delete",audio, category)
            os.remove(os.path.join(audio_path, audio))