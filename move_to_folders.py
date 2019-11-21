import csv
import glob
import shutil

class_label = {"/m/05tny_":"Bark","/m/01hsr_":"Sneeze","/m/0395lw":"Bell","/m/042v_gx":"Acoustic Guitar","/m/0ngt1":"Thunder","/m/014zdl":"Explosion","/m/03kmc9":"Siren","/m/01j3sz":"Laughter"}
in_list = list(csv.reader(open("D:/data/filtered_data_2000.csv","r")))

for filea in glob.glob("D:/data/*.wav"):

    file_id = filea.split("\\")[1].split(".wav")[0].split("_")[0]
    filename = filea.split("\\")[1]
    print(filename)
    #print(file_id)
    for item in in_list:
        if file_id == item[0]:
            #print(item[3].replace('"', '').replace(" ","").split(","))
            for a in item[3].replace('"', '').replace(" ","").split(","):
                try:
                    class_ID = class_label[a]
                    print(item, class_ID)
                    shutil.move(filea, "data/New Folder/"+class_ID+"/"+filename)
                    break
                except KeyError:
                    continue
        #break
