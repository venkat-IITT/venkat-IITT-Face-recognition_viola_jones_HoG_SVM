import cv2 
import numpy as np
import matplotlib as plt 
def viola_jones(grimg):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Detect faces in the image
    faces = face_cascade.detectMultiScale(grimg, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    x = faces[0]
    y = faces[1]
    w = faces[2]
    h = faces[3]
    cp_img = grimg[y:y+h, x:x+w]
    return cp_img
