import os
from tkinter import *
from tkinter import messagebox
from tkinter import *
import pymysql
from tkinter import messagebox, filedialog
import cv2
import mysql.connector as mysql

import matplotlib.pyplot as plt

from PIL import Image, ImageTk
import tkinter as tk
from cvlib.object_detection import draw_bbox

global act1
from signup import Signup
from login import Login
from main import ViewData
def Signupmeth():
    sign = Signup()


def Loginmeth():
    log = Login()


winadmin = Tk()
winadmin.title("COIFFURE")
winadmin.maxsize(width=1000, height=800)
winadmin.minsize(width=1000, height=800)
winadmin.configure(bg='#34bfbb')

image1 = Image.open("1.jpg")
img = image1.resize((1100, 1000))

test = ImageTk.PhotoImage(img)

label1 = tk.Label(winadmin, image=test)
label1.image = test

# Position image
label1.place(x=1, y=1)

# image1 = Image.open("3.png")
test = ImageTk.PhotoImage(img)

label1 = tk.Label(winadmin, image=test)
label1.image = test

# Create Canvas
# canvas1 = Canvas(win, width=400, height=400)

# canvas1.pack(fill="both", expand=True)

# Display image
# canvas1.create_image(0, 0, image=bg, anchor="nw")

Label(winadmin, text='COIFFURE', bg="#ffb366", font='verdana 15 bold') \
    .place(x=250, y=150)


btn_signup = Button(winadmin, text="Register", font='Verdana 10 bold', width="20", command=Signupmeth)
btn_signup.place(x=250, y=300)
btn_login = Button(winadmin, text="Login", font='Verdana 10 bold', width="20", command=Loginmeth)
btn_login.place(x=250, y=350)
btn_exit = Button(winadmin, text="Exit", font='Verdana 10 bold', width="20", command=quit)
btn_exit.place(x=250, y=400)
winadmin.mainloop()