import math, os, codecs, sys
import numpy as np
from PIL import Image

def flip(image):       #rewrite using L[:, i]
    a, b, c = np.shape(image)[:3]
    flpd = np.zeros((c, a, b), dtype='uint8')
    for loop in range(c) :
        for i in range(b) :
            for n in range(a) :
                flpd[loop,n,i]=int(image[n,i,loop])
    return flpd
def unflip(image):
    c, a, b = np.shape(image)[:3]
    flpd = np.zeros((a, b, c), dtype='uint8')
    for loop in range(c) :
        for i in range(b) :
            for n in range(a) :
                flpd[n,i,loop]=int(image[loop,n,i])
    return flpd

def grayscale(image) :
    return 0.2126*image[0] + 0.7152*image[1] + 0.0722*image[2]

def pool(image, size, type) :
    a, b = np.shape(image)[:2]
    conv=np.zeros((a-size+1, b-size+1))
    for l in range(a-size+1) :
        for n in range(b-size+1) :
            mtx=image[l][n:n+size]
            for i in range(l+1,l+size-1) :
                mtx=np.vstack((mtx, image[i][n:n+size]))
            if type == "max" :
                conv[l][n]=mtx.max()
            else :
                conv[l][n]=np.average(mtx)
    return conv

def convo(m1,m2, *args) : #convolution de deux matrices de dim 2, m1>=m2
    m1,m2 = np.array(m1), np.array(m2)
    ls = len(m1[0])-len(m2[0])+1
    hs = len(m1)-len(m2)+1
    s = np.zeros((hs, ls))
    for h in range(hs) :
        for l in range(ls) :
            mtx = m1[h][l:l+len(m2[0])]
            for i in range(1,len(m2)) :
                mtx = np.vstack((mtx,m1[h+i][l:l+len(m2[0])]))
            mtx = mtx * m2
            su = np.sum(mtx)
            if "prop" in args :
                if su<0 : su=0
                if su>255 : su=255
            s[h][l] = su
    return s

def kernel(image, pol, *args) : #mettre les poids du kernel en dernier argument

    image=flip(image)
    print(np.shape(image))
    image = np.round(grayscale(np.array(image))).astype('uint8')

    ker = []
    if 'gradient' in args :
        ker.append(args[-1])
    if 'r_sobel' in args :
        ker.append(np.array([[1,0,-1], [2,0,-2], [1,0,-1]]))
    if 'top_sobel' in args :
        ker.append(np.array([[1,2,1], [0,0,0], [-1,-2,-1]]))
    if 'outline' in args :
        ker.append(np.array([[ -1, -1, -1],[ -1, 8, -1],[ -1, -1, -1]]))
    if 'i' in args :
        ker.append(np.array([[ 0, 0, 0],[ 0, 1, 0],[ 0, 0, 0]]))
    if 'bot_sobel' in args :
        ker.append(np.array([[-1,-2,-1], [0,0,0], [1,2,1]]))
    print(np.shape(image))
    a, b = np.shape(image)[:2]

    conv=[]
    for loop in range(len(ker)) :
        print(loop)
        conv.append(convo(image, ker[loop], "prop"))

        if pol!=0 :
            for i in range(pol) :
                if 'max' in args :
                    sortie=pool(conv[loop], 5, 'max')
                else :
                    sortie=pool(conv[loop], 5, 'av')
            conv[loop]=sortie

    return np.round(conv).astype('uint8')




##tests
image = Image.open('F:\ppdp\ML_test1\quicky.JPG')
image = image.resize((256,144))#,Image.ANTIALIAS)
image.show()
data = np.asarray(image)
#data  = np.round(256/(1+np.exp(-0.035*(data-128)))) #augmentation contraste
data = kernel(data,0,'top_sobel', 'r_sobel', 'outline')
print(np.shape(data))
images=[Image.fromarray(data[i], 'L') for i in range(len(data))]
total_width, max_height=np.shape(data)[0]*np.shape(data)[2], np.shape(data)[1]
new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.show()





##current errors
#>>> (executing file "Kernel.py")
# (3, 720, 1280)
# (720, 1280)
# (5, 718, 1278) (720, 1280)
# 0
# Traceback (most recent call last):
#   File "F:\ppdp\ML_test1\Kernel.py", line 86, in <module>
#     data = kernel(data, 5,'bot_sobel','r _sobel','top_sobel','i', 'outline')
#   File "F:\ppdp\ML_test1\Kernel.py", line 74, in kernel
#     sortie[loop]=pool(conv[loop], 5, 'av')
# ValueError: setting an array element with a sequence.
#
# >>> (executing file "Kernel.py")
# (3, 720, 1280)
# (720, 1280)
# (5, 718, 1278) (720, 1280)
# 0
# Traceback (most recent call last):
#   File "F:\ppdp\ML_test1\Kernel.py", line 86, in <module>
#     data = kernel(data, 5,'bot_sobel','r_sobel','top_sobel','i', 'outline')
#   File "F:\ppdp\ML_test1\Kernel.py", line 74, in kernel
#     sortie[loop]=pool(conv[loop], 5, 'av')
# ValueError: setting an array element with a sequence.
#
