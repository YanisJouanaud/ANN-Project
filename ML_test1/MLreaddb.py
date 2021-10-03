import numpy as np
import os, codecs, sys
from PIL import Image
import random

def draw(tab) :
    #fonction dpour dessiner l'image dans le shell, non utilisé
    for i in range(len(tab)) :
        for n in range(len(tab[0])) :
            if tab[i][n]>190 :
                sys.stdout.write('☻')
            elif tab[i][n]>0 :
                sys.stdout.write('☺')
            else :
                sys.stdout.write('.')
        sys.stdout.write('\n')

def readdb():
    #fonction de création de dictionnaire contenant les exemples d'images
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #dir_path="F:\\programmation pour débile profond\\ML_test1"
    datapath = dir_path + '\\data'
    files = os.listdir(datapath)

    def get_int(b) :
        return int(codecs.encode(b, 'hex'),16)

    data_dict = {}
    for file in files :
        if file.endswith('ubyte') :
            with open (datapath+'\\'+file, 'rb') as f:
                data = f.read()
                type = get_int(data[:4])
                length= get_int(data[4:8])
                if type==2051 :
                    category = 'images'
                    num_rows = get_int(data[8:12])
                    num_cols = get_int(data[12:16])
                    parsed = np.frombuffer(data, dtype = np.uint8, offset = 16)
                    parsed = parsed.reshape(length, num_rows, num_cols)
                elif type==2049 :
                    category='labels'
                    parsed=np.frombuffer(data, dtype=np.uint8, offset=8)
                    parsed = parsed.reshape(length)
                if length==10000 :
                    set = 'test'
                elif length==60000 :
                    set= 'train'
                data_dict[set+'_'+category] = parsed
    return data_dict




def readcats() :

    dir_path = os.path.dirname(os.path.realpath(__file__))
    #dir_path="F:\\ppdp\\ML_test1\\reseau"
    with open(dir_path + '\\list.txt', 'r') as f :
        txt=f.read()
    txt=txt.split("\n")
    txt=txt[6:]
    random.shuffle(txt)
    txt = [txt[i].split(" ")[0] for i in range(len(txt)) if txt[i][0].isupper()]
    data=[]
    for i in range(len(txt)) :
        image = Image.open(dir_path + "\\" + txt[i] + ".jpeg")
        data.append(np.asarray(image))

    return [txt, data]













