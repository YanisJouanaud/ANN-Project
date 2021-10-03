import math, sys, os
import numpy as np
from save import load()
from Kernel import kernel, grayscale
from guesser import guess

def catin(image):
    seuil=0.9
    w, s = load("catsws.txt")
    image=kernel(image, "top_sobel", "outline", "i")
    neuro=guess(image, nn[1:], w, s)
    if neuro[-1][0] < seuil :
        return False
    else :
        return True

def quickyin(image):
    seuil = 0.8
    w,s = load("quickyws.txt")
    image=kernel(image, "top_sobel", "outline", "i")
    neuro=guess(image, nn[1:], w, s)
    if neuro[-1][0] < seuil :
        return False
    else :
        return True



