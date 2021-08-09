import os, glob
import cv2
import shutil

#directory
from os import listdir, makedirs
from os.path import isfile, join

import pathlib

#memperoleh dataset untuk melakukan preprocessing
def dataset():
    path_data = 'full_dataset'
    outputdir = 'output_dataset'

    # membuat output direktori
    try:
        makedirs(outputdir)
    except:
        print('Error: Directory sudah ada!')

dataset()

#GRAYSCALE
def preprocessing(path_data='full_dataset', outputdir='output_dataset'):
    # unused
    files = list(filter(lambda f: isfile(join(path_data, f)), listdir(path_data)))

    # Here src_path is the location where images are saved.
    for filenames in files:
         try:
            img = cv2.imread(os.path.join(path_data, filenames))  # membaca gambar yang ada di direktori
            img = cv2.resize(img, (200, 200))  # memperkecilkan gambar
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # konversi menjadi grayscale
            hasil = join(outputdir, filenames)
            cv2.imwrite(hasil, gray)

            print('{} telah berhasil dipre-processing'.format(filenames))
         except:
            print("{} tidak bisa dikonversi!".format(filenames))

preprocessing()


print()