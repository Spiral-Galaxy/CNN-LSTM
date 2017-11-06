import os
import cv2
import os.path
from Config import Video_Path


All_Class = {}
Class_Names = os.listdir(Video_Path)

for i in range(len(Class_Names)):
    current = Class_Names[i]
    All_Class[current] = os.listdir(os.path.join(Video_Path, current))

print(All_Class)
print(Class_Names)

v = cv2.VideoCapture(os.path.join(Video_Path,Class_Names[0],All_Class[Class_Names[0]][0]))
print(v.isOpened())

