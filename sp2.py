import cv2 
import numpy as np
import matplotlib as plt 

# Load the pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('s10_01.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cp_img = gray_image[y:y+h, x:x+w]

# Display the image with detected faces
cv2.imshow('Detected Faces', cp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

face_regions = []
for (x, y, w, h) in faces:
   face_regions.append(gray_image[y:y+h, x:x+w])




