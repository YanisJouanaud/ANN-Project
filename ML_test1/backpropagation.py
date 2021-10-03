import numpy as np
import math
from Kernel import convo

def sig(x) :
    #fonction sigmoid, non utilisé au final
    if type(x)=='numpy.ndarray' :
        return 1/(1+np.exp(-1*x))
    if type(x)=='int' :
        return 1/(1+math.exp(-1*x))


def backprop(neur, y, w, b, nn, *args):
#fonction de retropropagation du gradient pour que le réseau "apprenne"
#neur = réseau après travail ; y = bonne réponse ; w,b = weight et biais actuel
#nn = format du réseau neuronal
#args : "conv" et l'image im si convolution


    cneur=np.array([-5000*np.exp(np.zeros((nn[i],1))) for i in range(len(nn))])

    def cneurcal(l, j) :
        if cneur[l+1][j][0]!=-5000 :
            sum = cneur[l+1][j][0]
        elif l==len(neur)-2 :
            sum = 2*(neur[l+1][j][0]-y[j][0])
            cneur[l+1][j][0] = sum
        else :
            sum = 0
            k = j
            for j in range(len(neur[l+2])) :
                sum += w[l+1][j][k]*neur[l+2][j][0]*(1-neur[l+2][j][0])*cneurcal(l+1, j)
            cneur[l+1][k][0]=sum
        return sum


    if "conv" in args :
        long=list(range(len(nn)-1))
        long.pop(0)
        cw=np.array([np.zeros((nn[i+1],nn[i])) for  i in long])
        cb=np.array([np.zeros((nn[i+1],1)) for  i in long])
        lst=list(reversed(range(len(neur)-1)))
        lst.pop()
        for l in lst:
            for j in range(len(cb[l])) :
                for k in range(len(cneur[l])) :
                    cw[l][j][k]=neur[l][k][0]*neur[l+1][j][0]*(1-neur[l+1][j][0])*cneurcal(l, j)
                cb[l][j][0]=neur[l+1][j][0]*(1-neur[l+1][j][0])*cneurcal(l, j)
        k=(np.shape(im)[1]+np.shape(im)[2]-sqrt((np.shape(im)[1]+np.shape(im)[2])^2-4*(np.size(im)-np.size(cneur[0]))))/2
        ck=convo(args[1], cneur[0].reshape((np.shape(im)[1]-k, np.shape(im)[2]-k)))
        return cw, cb, ck

    else :
        cw=np.array([np.zeros((nn[i+1],nn[i])) for  i in range(len(nn)-1)])
        cb=np.array([np.zeros((nn[i+1],1)) for  i in range(len(nn)-1)])
        for l in reversed(range(len(neur)-1)) :
            for j in range(len(cb[l])) :
                for k in range(len(cneur[l])) :
                    cw[l][j][k]=neur[l][k][0]*neur[l+1][j][0]*(1-neur[l+1][j][0])*cneurcal(l, j)
                cb[l][j][0]=neur[l+1][j][0]*(1-neur[l+1][j][0])*cneurcal(l, j)

        return cw, cb