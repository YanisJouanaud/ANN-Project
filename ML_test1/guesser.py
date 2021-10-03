import numpy as np
import math


def init_wb(nn) :
    #création weight et bias au hasard lors du premier entrainement
    w, b = [], []
    low, high = -100 , 100
    for i in range(len(nn)-1) :
        np.random.randint2 = lambda *args, dtype=np.float64: np.random.randint(*args).astype(dtype)
        x = np.random.randint2(low, high, (nn[i+1], nn[i]), dtype='i4')/high
        w.append(x)
        b.append(np.zeros((nn[i+1], 1)))
    return w,b





def guess(image, neurs, w, b) :
    #fonction de propagation au sein du réseau

    #image : array numpy 28x28 8bits, image=readddb().get(set_category)[x]
    #neurs : tuple contenant le nombre de neurone de chaque layer
    #w : liste contenant array de weights de chaque layer
    #b : liste contenant listes de bias de chaque layer
    #neuro : array of the final values took by the neurones, 1 line = 1 layer



    image=image.reshape(np.size(image),1)
    image=image/255
    neuro=[image]
    for i in range(len(neurs)) :
        n=np.array(np.dot(w[i], neuro[i]) + b[i], dtype=np.float64)
        for l in range(np.size(n)) :
            if n[l][0]<-700 :
                n[l][0]=-700
        n=1/(1+np.exp(-1*n))
        neuro.append(n)


    return np.array(neuro)



