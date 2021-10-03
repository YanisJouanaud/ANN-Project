#affichage
from tkinter import *
import time
import numpy as np
import os, codecs, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path :
    sys.path.append(dir_path)
from MLreaddb import readdb, draw
from guesser import guess, init_wb
from backpropagation import backprop
from reseau.save import save, load
import random

def affichagetest(img, neuro, guess) :
    #affichage de test de la propagation, non utilisé
    window =Tk()
    window.title("Hand Written Digits NN Recognition")
    window.geometry('1080x720')
    window.minsize(640, 480)
    window.config(background='#776767')
    frame= Frame(window, bg='#776767')
    left_frame = Frame(frame, bg='#776767')
    bottom_frame = Frame(frame, bg='#776767')
    right_frame = Frame(frame, bg='#776767')

    taille=2
    image=Canvas(left_frame, bg="#776767", confine=True, height=280*taille, width=280*taille)
    l=10*taille
    for y in range(28) :
        for x in range(28) :
            a='#'+hex(img[y][x])[2:]*3
            image.create_oval(x*l, y*l , x*l+l, y*l+l, fill=a, outline='black', state='disabled')
    image.pack(expand=1, anchor='center', padx=20)
    left_frame.grid(row=0, column=0, sticky='nesw')


    output=Canvas(right_frame, bg='#776767', confine=True, height=280*taille, width=90*taille)
    l=28*taille
    for i in range(10):
        a='#'+hex(255-int(neuro[i][0]*255))[2:]*3
        output.create_oval(5*taille, i*l, 33*taille, i*l+l, fill=a, outline='black', state='disabled')
        output.create_text(65*taille, i*l+14*taille, text=str(int(1000*neuro[i][0])/10)+'%', font=('Comic sans', 20))
    output.pack(expand=1, anchor='center', padx=20)

    right_frame.grid(row=0, column=1, sticky='nesw')


    resultat = Entry(bottom_frame, font=('Comic sans', 20), bg='#776767', fg ='white', takefocus=0)
    resultat.pack(side='bottom', pady=20)
    resultat.insert(0, guess)
    bottom_frame.grid(row=1,column=0, sticky='s')
    frame.pack(expand=1, anchor='center')

    window.mainloop()

def init_affichage() :
    #fonction d'affichage de test, non fini
    n=[0, 0]
    def computerand() :
         #fonction de test du réseau avec une image d'entrée au hasard
        output.delete('pix')
        image.delete('pix')
        x=random.randint(0,10000)
        dessin=readdb().get("test_images")[x]
        rep=readdb().get("test_labels")[x]
        l=20
        taille=2
        for y in range(28) :
            for x in range(28) :
                a='#'+hex(dessin[y][x])[2:]*3
                image.create_oval(x*l, y*l , x*l+l, y*l+l, fill=a, outline='black', state='disabled', tag='pix')
        neuro=guess(dessin, (16, 16, 10), w, b)
        l=55
        for i in range(10):
            a='#'+hex(255-int(neuro[-1][i][0]*255))[2:]*3
            output.create_oval(5*taille, i*l, 33*taille, i*l+l, fill=a, outline='black', state='disabled', tag='pix')
            output.create_text(65*taille, i*l+14*taille, text=str(int(1000*neuro[-1][i][0])/10)+'%', font=('Comic sans', 20), tag='pix')
        num=np.argmax(neuro[-1])
        resultat.delete(0, END)
        resultat.insert(0, num)
        n[0]+=1
        if num==rep :
            n[1]+=1
        reussite.delete('pix')
        reussite.create_text(40,40, text=str(int(100*n[1]/n[0]))+' %', font=('Comic sans', 20), tag='pix')

        window.update()

    def compute() :
        pass


    def drawing() :
        pass

    def clear():
        l=20
        image.delete('pix')
        for y in range(28) :
            for x in range(28) :
                image.create_oval(x*l, y*l , x*l+l, y*l+l, fill='black', outline='black', state='disabled', tag='pix')
        dessin=np.zeros((28,28))
        resultat.delete(0, END)
        window.update()


    print('lancé ! ')

    w, b = load("weights_biases.txt")
    window =Tk()
    window.title("Hand Written Digits NN Recognition")
    window.geometry('1080x720')
    window.minsize(640, 480)
    window.config(background='#776767')
    frame= Frame(window, bg='#776767')
    left_frame = Frame(frame, bg='#776767')
    bottom_frame = Frame(frame, bg='#776767')
    right_frame = Frame(frame, bg='#776767')
    middle_frame = Frame(frame, bg='#776767')

    taille=2
    image=Canvas(left_frame, bg="#776767", confine=True, height=280*taille, width=280*taille)
    image.pack(expand=1, anchor='center', padx=20)
    Clear = Button(left_frame,  text='Clear', font=('Comic sans', 20), bg='white', fg ='#776767', command=clear)
    Clear.pack(expand=1, anchor='center')
    left_frame.grid(row=0, column=0, sticky='nesw')

    shuffle = Button(middle_frame, text='Shuffle', font=('Comic sans', 20), bg='white', fg ='#776767', command=computerand)
    shuffle.pack(expand=1, anchor='center')
    middle_frame.grid(row=0, column=1, sticky='nesw')

    output=Canvas(right_frame, bg='#776767', confine=True, height=280*taille, width=90*taille)
    output.pack(expand=1, anchor='center', padx=20)
    right_frame.grid(row=0, column=2, sticky='nesw')


    resultat = Entry(bottom_frame, font=('Comic sans', 20), bg='#776767', fg ='white', takefocus=0)
    resultat.pack(side='left', pady=20)
    reussite = Canvas(bottom_frame, bg='#776767', height=80, width=80)
    reussite.pack(side='left', anchor='se', padx=50)
    bottom_frame.grid(row=1,column=0, sticky='s')
    frame.pack(expand=1)
    computerand()
    window.mainloop()

