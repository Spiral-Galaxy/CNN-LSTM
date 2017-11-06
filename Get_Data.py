import tensorflow as tf
import cv2
import numpy as np
from Config import *
from Get_Name import All_Class, Class_Names


def get_seq_from_images(images):
    seq = []
    for i in range(images.shape[0]):
        seq.append(images[i,:,:,:])
    return seq


def Load_Videos(path):
    V_d_path = path + '/Biking'
    V_fpath = V_d_path + '/v_Biking_g01_c01.avi'
    print(V_fpath)
    v = cv2.VideoCapture(V_fpath)
    print(v.isOpened())
    ret = 1
    images = []
    while ret == 1:
        ret, image = v.read()
        if ret == True:
            s = (224,224,1)
            img1 = cv2.resize((image / 255).astype(np.float32)[:, :, 0], ( 224, 224))
            img2 = cv2.resize((image / 255).astype(np.float32)[:, :, 1], (224, 224))
            img3 = cv2.resize((image / 255).astype(np.float32)[:, :, 2], (224, 224))
            image = np.concatenate((img1.reshape(s), img2.reshape(s)), axis = 2)
            image = np.concatenate((img3.reshape(s), image), axis=2)
            images.append(image)
    v.release()

    return images

def Trans_Images_to_Array(images,s):
    array = np.concatenate((images[0].reshape(s),images[1].reshape(s)),0)
    for i in range(2,len(images)):
        array = np.concatenate((array, images[i].reshape(s)),0)
    return array


def Get_Data(Video_Path):
    images = Load_Videos(Video_Path)
    print('images')
    print(images)
    s = [-1, 224, 224, C_size]

    return Trans_Images_to_Array(images, s)
