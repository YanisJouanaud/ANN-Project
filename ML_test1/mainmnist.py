#Ceci est un algorithme de machine learning avce une interface
#
#Je laisse intentionnellement bon nombre de fonctions de tests en cas où.
#
#Pour entrainer le réseau vous pouvez comenter/décommenter les deux dernières lignes,
#lancez le programme, et écrire 'train' dans le Shell (sans les guillemets).
#
#UI toujours en cours d'amélioration, notamment avec une fonction de dessin prévu.
#
#Disfonctionnement possible au niveau des modules, je sais pas vraiment
#comment ça marche quand on partage à ce niveau.
#
#J'espere que le tout reste lisible, tout a été ecris de A à Z sans copié-collé,
#et ce fut un vrai casse-tête, notamment pour la descente de gradient.
#
#Depuis ici : ctrl + shift + E pour lancer le programmme
from multiprocessing import Process
import numpy as np
import os, codecs, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path :
    sys.path.append(dir_path)
    sys.path.append(dir_path+"\\reseau")
from MLreaddb import readdb, draw
from guesser import guess, init_wb
from affichage import init_affichage, affichagetest
from backpropagation import backprop
from reseau.save import save, load
from Kernel import kernel, grayscale, convo
import random

def init() :

    nn = [784, 16, 16, 10]
    bs = 150
    alpha=0.1
    seuil=0.05
    return nn, bs, alpha, seuil

def train(tpl) :
    #fonction d'entrainement ; pour reprendre l'entrainement à zero,
    #commenter/décommenter les deux lignes suivantes
    nn, bs, alpha, seuil = tpl
    w, b = init_wb(nn)
    #w, b = load()
    data=readdb()
    print(type(data))
    k=0
    nb=0
    cost=10
    while cost/bs>seuil :
        cost=0
        result=0
        gradientw=np.array([np.zeros((nn[i+1], nn[i])) for i in range(len(nn)-1)])
        gradientb=np.array([np.zeros((nn[i+1],1)) for i in range(len(nn)-1)])
        for i in range(bs) :
            #x=np.random.randint(60000, size=1)[0]
            image=data.get('train_images')[(k*bs+i)%50000]
            neuro=guess(image, nn[1:], w, b)
            rep=data.get('train_labels')[(k*bs+i)%50000]
            y=np.zeros((10,1))
            y[rep][0]=1
            grad=backprop(neuro, y, w, b, nn)
            gradientw+=grad[0]
            gradientb+=grad[1]
            cost+=np.sum((neuro[-1]-y)**2)
            if np.argmax(neuro[-1])==rep :
                result+=1
        result/=bs
        k+=1
        print(result, '//', cost/bs, '//', k*bs, '//', k) #taux de réussite // loss rate // nombre d'entrainement // epoch
        gradientw/=bs
        gradientb/=bs
        if cost/bs>seuil :
            w-=(gradientw*alpha)
            b-=(gradientb*alpha)
        prev=cost
    save(w, b, "\\weights_biasis.txt")

def main() :
    consigne=input("train ou test ?\n")
    if consigne=='test' :
        init_affichage()
    if consigne=='train' :
        train(init())



def test() :
    #fonction de test de la propagation, non utilisé
    w, b = init_wb(nn)
    #w, b = load()
    x=np.random.randint(5000, size=1)[0]
    image=readdb().get('train_images')[x]
    neuro=guess(image, (16, 16, 10), w, b)

    print('\n\n\n\n')
    print(neuro[-1])
    print(np.sum((neuro[-1]-np.zeros((10,1)))))
    print('\n\n')
    print('guess : ', np.argmax(neuro[-1]))
    affichagetest(image, neuro[-1], np.argmax(neuro[-1]))



train(init())






