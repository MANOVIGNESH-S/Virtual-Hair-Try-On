from tkinter import *
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk 
import random
from mysql.connector import connection, cursor
import mysql
import mysql.connector
import mysql.connector as mysql
from login import Login 

class Signup:
        def __init__(self):
                
                # signup database connect 
                def action():
                        if userid.get()=="" or username.get()=="" or address.get()=="" or emailid.get()=="" or mobile.get()=="" or password.get()=="":

                        
                                messagebox.showerror("Error" , "All Fields Are Required" , parent = winsignup)                     
                        else:                                           
                                try:
                                        db_connection = mysql.connect(user='root', password='root', host='127.0.0.1',
                                                                      charset='utf8',
                                                                      database='hairstyle')
                                        cur = db_connection.cursor()
                                        cur.execute("select * from user_information where username='%s'"%username.get())
                                        row = cur.fetchone()
                                        if row!=None:
                                                messagebox.showerror("Error" , "User Name Already Exits", parent = winsignup)
                                        else:
                                                cur.execute("insert into user_information(username,password,address,emailid,mobile) values (%s,%s,%s,%s,%s)",
                                                        ( 
                                                        username.get(),
                                                        password.get(),
                                                        address.get(),
                                                        emailid.get(),
                                                        mobile.get(),
                                                        
                                                        ))
                                                db_connection.commit()
                                                db_connection.close()
                                                
                                        messagebox.showinfo("Success" , "Registered Successfully" , parent = winsignup)
                                        clear()                                        
                                        
                                except Exception as es:
                                        messagebox.showerror("Error" , f"Error Due to : {str(es)}", parent = winsignup)

                # close signup function			
                def switch():
                        winsignup.destroy()
                        login= Login()

                # clear data function
                def clear():
                        userid.delete(0,END)
                        username.delete(0,END)
                        address.delete(0,END)
                        emailid.delete(0,END)
                        mobile.delete(0,END)
                        password.delete(0,END) 
                
                        
                        
                # start Signup Window	

                winsignup = Toplevel()
                winsignup.title("Hair Style Detection")
                winsignup.maxsize(width=1000, height=800)
                winsignup.minsize(width=1000, height=800)
                winsignup.configure(bg='#34bfbb')

                image1 = Image.open("3.jpg")
                img = image1.resize((1100, 1000))

                test = ImageTk.PhotoImage(img)

                label1 = tk.Label(winsignup, image=test)
                label1.image = test

                # Position image
                label1.place(x=1, y=1)

                # image1 = Image.open("3.png")
                test = ImageTk.PhotoImage(img)

                label1 = tk.Label(winsignup, image=test)
                label1.image = test

                # Display image
                # canvas1.create_image(300, 300, image=bg, anchor="nw")
                Label(winsignup, text='User Registration Process', bg="#ffb366", font='verdana 15 bold') \
                        .place(x=250, y=50)


                #heading label
                heading = Label(winsignup ,text = "Register" , font = 'Verdana 15 bold')
                heading.place(x=80 , y=60)

                # form data label
                userid = Label(winsignup,   text= "User ID :" , font='Verdana 10 bold')
                userid.place(x=80,y=130)

                username = Label(winsignup,  text= "User Name :" , font='Verdana 10 bold')
                username.place(x=80,y=160)

                password = Label(winsignup,   text= "Password :" , font='Verdana 10 bold')
                password.place(x=80,y=190)

                address = Label(winsignup,   text= "Address :" , font='Verdana 10 bold')
                address.place(x=80,y=260)


                emailid = Label(winsignup, text= "Email ID :" , font='Verdana 10 bold')
                emailid.place(x=80,y=290)
         
                mobile = Label(winsignup,  text= "Mobile :" , font='Verdana 10 bold')
                mobile.place(x=80,y=320)

                # Entry Box ------------------------------------------------------------------

                uid = StringVar()
                username = StringVar()
                password=StringVar()
                address = StringVar()
                emailid= StringVar()
                mobile= StringVar() 

                uid = ""
                print(uid)
                pintext = StringVar()
                pintext.set(uid)
                
                userid = Entry(winsignup, width=40, text = ""  )
                userid.place(x=200 , y=130)

                db_connection = mysql.connect(user='root', password='root', host='127.0.0.1', charset='utf8',
                                              database='hairstyle')
                cur = db_connection.cursor()
                
                cur.execute("select MAX(userid) from user_information")
                row = cur.fetchone()
                uid = row[0] 
                if uid == None:
                        uid=0                
                uid=uid+1
                userid.delete(0, tk.END)
                userid.insert(0, uid)
                
                username = Entry(winsignup, width=40 , textvariable = username)
                username.focus()
                username.place(x=200 , y=160)
                
                password = Entry(winsignup, width=40,  show='*',  textvariable = password)
                password.place(x=200 , y=190)
                
                address = Entry(winsignup, width=40, textvariable=address)
                address.place(x=200 , y=260)
         

                emailid = Entry(winsignup, width=40,textvariable = emailid)
                emailid.place(x=200 , y=290)

                mobile = Entry(winsignup, width=40 , textvariable = mobile)
                mobile.place(x=200 , y=320)
                   
                # button login and clear

                btn_signup = Button(winsignup, text = "Register" ,font='Verdana 10 bold', command = action)
                btn_signup.place(x=200, y=360)


                btn_login = Button(winsignup, text = "Clear" ,font='Verdana 10 bold' , command = clear)
                btn_login.place(x=280, y=360)


                sign_up_btn = Button(winsignup , text="Login",font='Verdana 10 bold'  , command = switch )
                sign_up_btn.place(x=350 , y =360)


                winsignup.mainloop()

