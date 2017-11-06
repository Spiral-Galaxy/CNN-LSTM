import numpy as np
import cv2

cap = cv2.VideoCapture('/home/jiaweihuang/Program/PyCharm/hybrid/bike.avi')

# take first frame of the video

print(cap.isOpened())

ret, frame = cap.read()

