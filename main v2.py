#import library untuk pembuatan aplikasi GUI
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import tkinter.font as tkFont

#untuk array
import numpy as np

#import library untuk foto
from PIL import Image, ImageTk
import cv2

#import library untuk sistem operasi
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model

model = load_model('leafClassification.h5', compile=False)  # load model
global height, width, channel
#model.summary()

load_input = model.input
input_shape = list(load_input.shape)
height = int(input_shape[1])
width = int(input_shape[2])
channel = int(input_shape[3])

#print(height, width, channel)

root = Tk()
root.title("Aplikasi Klasifikasi Jenis Tanaman")
root.geometry("920x550")
root.resizable(False,False)

file_daun = tk.StringVar()
test_result_var = tk.StringVar()
accuracy_result_var = tk.StringVar()

def browse_jpg():
    global filename, img, file_daun
    filename = filedialog.askopenfilename(
        initialdir=os.getcwd(), title="Pilih file daun jpg",
        filetypes=(("JPG files", "*.jpg"), ("all files", "*.*")))
    file_daun.set(filename)
    txt_filejpg.delete("1.0", tk.END)
    txt_filejpg.insert('end', file_daun.get())
    inputValue = txt_filejpg.get("1.0", "end-1c")
    img = Image.open(filename)
    img = img.resize((200,200))
    tkimage = ImageTk.PhotoImage(img)
    daun_img.configure(image=tkimage)
    daun_img.image = tkimage

def load_img():
    path = file_daun.get()
    global imgs
    imgs = cv2.imread(os.path.join(path))
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    imgs = cv2.resize(imgs, (height, width))
    messagebox.showinfo("Message", "Image loaded!")
    return

def predict():
    labels = ['Phyllostachys Edulis', 'Chrysobalanaceae', 'Cercis Chinensis', 'Indigofera Tinctoria L']
    predict_image = tf.keras.preprocessing.image.img_to_array(imgs)
    predict_image = (np.array(predict_image))/255
    predict_image = np.expand_dims(predict_image, axis=0)
    predict_image = np.vstack([predict_image])
    pred = model.predict(predict_image)
    print(pred)
    accuracy_text = "Akurasi: {:.2f}" .format(100 * np.max(pred))
    result_text = "Predicted class: " + str(labels[np.argmax(pred)])

    test_result_var.set(result_text)
    text_kelas.delete("1.0", tk.END)
    text_kelas.insert('end', test_result_var.get())
    inputValue = text_kelas.get("1.0", "end-1c")

    accuracy_result_var.set(accuracy_text)
    text_akurasi.delete("1.0", tk.END)
    text_akurasi.insert('end', accuracy_result_var.get())
    inputValue = text_akurasi.get("1.0", "end-1c")
    return pred


fontTitle = tkFont.Font(size=40)
lbl_judul = tk.Label(root, font=fontTitle, text="LEAFIFY") #label judul
lbl_judul.place(x=180, y=150, width=230, height=60)

lbl_filejpg = tk.Label(root, text="File Gambar ") #label
lbl_filejpg.place(x=50, y=250, width=100, height=25)

txt_filejpg = tk.Text(root) #text input state="disabled"
txt_filejpg.place(x=150, y=252, width=280, height=20)

btn_filejpg = tk.Button(root, text="Browse", command=browse_jpg) #browse button
btn_filejpg.place(x=450, y = 250, height=25)

fontButton = tkFont.Font(size=13)
btn_load = tk.Button(root, font=fontButton, text="Load Image", command=load_img) #load image button
btn_load.place(x=160, y = 310, width=120, height=50)

btn_predict = tk.Button(root, font=fontButton, text="Prediksi", command=predict) #predict button
btn_predict.place(x=320, y = 310, width=100, height=50)

#canvas
daun_img = tk.Label(root, bg='gray', borderwidth=1, relief="solid")
daun_img.place(x=600, y=100, width=200, height=200)

#text label "Hasil prediksi"
fontTitle2 = tkFont.Font(size=14)
lbl_hasil = tk.Label(root, font=fontTitle2, text="Hasil prediksi:") #label judul
lbl_hasil.place(x=590, y=310, width=230, height=60)

text_kelas = tk.Text(root, font=tkFont.Font(size=9))
text_kelas.place(x=595, y=370, width=210, height=20)
text_kelas.insert('end', test_result_var.get())

text_akurasi = tk.Text(root, font=tkFont.Font(size=9))
text_akurasi.place(x=595, y=410, width=210, height=20)
text_akurasi.insert('end', accuracy_result_var.get())

#mulai program
root.mainloop()