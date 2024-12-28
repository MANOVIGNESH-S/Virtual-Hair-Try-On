from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import random
import mysql.connector
from mysql.connector import connection, cursor,connect
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import mysql
import mysql.connector
import mysql.connector as mysql
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# from  Landdetails import LandDet
# from viewdetails import ViewLand
from main import ViewData


class Login:
    def __init__(self):

        def checklogin():
            if useridentry.get() == "" and passwordentry.get() == "":
                messagebox.showerror("Error", "Enter User id", parent=winlogin)
            elif useridentry.get() == "Admin" and passwordentry.get() == "Admin":
                land = ViewData()
            else:
                try:
                    db_connection = mysql.connect(user='root', password='root', host='127.0.0.1', charset='utf8',
                                                  database='hairstyle')
                    cur = db_connection.cursor()

                    cur.execute("select username,password from user_information where username=%s and password=%s ",
                                (useridentry.get(), passwordentry.get()))
                    row = cur.fetchone()
                    land = ViewData()

                    if row == None:
                        messagebox.showerror("Error", "Invalid User id", parent=winlogin)

                    else:
                        messagebox.showinfo("Success", "Successfully Login", parent=winlogin)
                        db_connection.close()
                        # exec(open("Test1.py").read())

                except Exception as es:
                    messagebox.showerror("Error", f"Error Duo toooo : {str(es)}", parent=winlogin)


        winlogin = Toplevel()
        winlogin.title("Hair Style Detection")
        winlogin.maxsize(width=1000, height=800)
        winlogin.minsize(width=1000, height=800)
        winlogin.configure(bg='#34bfbb')

        image1 = Image.open("2.jpg")
        img = image1.resize((1100, 1000))

        test = ImageTk.PhotoImage(img)

        label1 = tk.Label(winlogin, image=test)
        label1.image = test

        # Position image
        label1.place(x=1, y=1)

        # image1 = Image.open("3.png")
        test = ImageTk.PhotoImage(img)

        label1 = tk.Label(winlogin, image=test)
        label1.image = test

        # Display image
        #canvas1.create_image(300, 300, image=bg, anchor="nw")
        Label(winlogin, text='User Login Process', bg="#ffb366", font='verdana 15 bold') \
            .place(x=180, y=50)




        # form data label
        userid = Label(winlogin, text="User Name :", font='Verdana 10 bold')
        userid.place(x=80, y=130)

        # form data label
        password = Label(winlogin, text="Password :", font='Verdana 10 bold')
        password.place(x=80, y=180)

        # Entry Box
        userid = StringVar()
        password = StringVar()
        useridentry = Entry(winlogin, width=40, textvariable=userid)
        useridentry.focus()
        useridentry.place(x=200, y=130)

        passwordentry = Entry(winlogin, width=40, show='*', textvariable=password)
        passwordentry.focus()
        passwordentry.place(x=200, y=180)

        # button login and clear

        btn_login = Button(winlogin, text="Login", font='Verdana 10 bold', command=checklogin)
        btn_login.place(x=200, y=240)

        btn_exit = Button(winlogin, text="Exit", font='Verdana 10 bold', command=quit)
        btn_exit.place(x=300, y=240)

        winlogin.mainloop()
