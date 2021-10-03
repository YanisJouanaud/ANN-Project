import sys
import numpy as np
import os

def size(tab) :
    #fonction servant à determiner la taille d'une matrice
    #contenant plusieurs matrices, non utilisé
    sum=0
    for i in range(np.size(tab)) :
        sum+=np.size(tab[i])
    return sum

def save(tab1, tab2, name) :
    #fonction de sauvegarde du réseau une fois l'entrainement terminé
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #dir_path="F:\\programmation pour débile profond\\ML_test1\\reseau"
    with open(dir_path + name, "w+") as f :
        for x in range(len(tab1)) :
            for y in range(len(tab1[x])) :
                for z in range(len(tab1[x][y])) :
                    f.write(str(tab1[x][y][z])+' ')
                f.write('//')
            f.write('\n')
        f.write("###\n")
        for x in range(len(tab2)) :
            for y in range(len(tab2[x])) :
                f.write(str(tab2[x][y][0])+' ')
            f.write('\n')





def load(name) :
    #fonction de chargements du réseau une fois entrainé
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #dir_path="F:\\programmation pour débile profond\\ML_test1\\reseau"
    with open(dir_path + "\\" + name, 'r') as f :
        txt=f.read()
    w=txt.split('###\n')[0]
    b=txt.split('###\n')[1]
    w=w.split('\n')
    b=b.split('\n')
    for i in range(len(w)) :
        w[i]=w[i].split('//')
        for n in range(len(w[i])) :
            w[i][n]=list(map(float, w[i][n].split()))
        w[i].pop()
        w[i]=np.array(w[i])
    for i in range(len(b)) :
        b[i]=np.array(list(map(float, b[i].split())))
        b[i]=b[i].reshape(np.size(b[i]), 1)
    b.pop()
    w.pop()
    b=np.array(b)
    w=np.array(w)
    return w, b



#print(os.listdir(os.path.dirname(os.path.realpath(__file__))+r'\images'))