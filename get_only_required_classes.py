import csv

class_label = {"/m/05tny_":"Bark","/m/01hsr_":"Sneeze","/m/0395lw":"Bell","/m/042v_gx":"Acoustic Guitar","/m/0ngt1":"Thunder","/m/014zdl":"Explosion","/m/03kmc9":"Siren","/m/01j3sz":"Laughter"}
counts = {"/m/05tny_":0,"/m/01hsr_":0,"/m/0395lw":0,"/m/042v_gx":0,"/m/0ngt1":0,"/m/014zdl":0,"/m/03kmc9":0,"/m/01j3sz":0}
reqd_classes = ["/m/01hsr_","/m/05tny_","/m/0395lw","/m/042v_gx","/m/0ngt1","/m/014zdl","/m/03kmc9","/m/01j3sz"]
in_list = list(csv.reader(open("unbalanced_train_segments.csv","r")))
print(len(in_list))
#print(in_list)
out_list = []
for b in in_list:
    for class_label in reqd_classes:
        try:
            if class_label in b[3] and counts[class_label] <= 2000:
                out_list.append(b)
                counts[class_label]+=1
        except IndexError as ie:
            continue
print(len(out_list), counts)
with open("filtered_data_2000.csv","w") as outfile:
    for items in out_list:
        outfile.write(",".join(items))
        outfile.write("\n")
